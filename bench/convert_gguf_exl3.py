#!/usr/bin/env python3
"""Stream GGUF weights into llmopt's EXL3 mul1 archive using bounded Metal requests.

This is approximate requantization with deterministic random signs and per-output
RMS scales, without calibration or Hessian compensation. The encoder is a
lookahead heuristic, not ExLlamaV3's optimizer. Dense embeddings/norms/routers
remain F16/F32; all aligned projection and expert matrices become EXL3_4.
"""
from __future__ import annotations
import argparse, hashlib, json, os, selectors, struct, subprocess, time
from pathlib import Path
import numpy as np
import gguf
from llmopt_backend.tensor_archive import TensorSpec, write_payloads


def fwht128(x, axis):
    moved = np.ascontiguousarray(np.moveaxis(x, axis, -1))
    shape = moved.shape
    v = moved.reshape(-1, 128)
    for step in (1, 2, 4, 8, 16, 32, 64):
        z = v.reshape(-1, 128 // (2 * step), 2, step)
        a, b = z[:, :, 0].copy(), z[:, :, 1].copy()
        z[:, :, 0], z[:, :, 1] = a + b, a - b
    v *= np.float32(1 / np.sqrt(128))
    return np.moveaxis(v.reshape(shape), -1, axis)


def specs_from_archive(path):
    with path.open('rb') as f:
        magic, version, flags, count, _ = struct.unpack('<8sHHIQ', f.read(24))
        if (magic, version, flags) != (b'LLMOPTWT', 1, 0): raise ValueError('archive header')
        for _ in range(count):
            size, tag, rank, reserved = struct.unpack('<IBBH', f.read(8))
            name = f.read(size).decode(); shape = struct.unpack(f'<{rank}Q', f.read(8 * rank))
            offset, length = struct.unpack('<QQ', f.read(16))
            yield TensorSpec(name, tag, shape, length, offset)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--extra-archive', type=Path)
    p.add_argument('--only', help='Exact tensor name for a bounded conversion probe')
    p.add_argument('--chunk-rows', type=int, default=1024)
    a = p.parse_args()
    if a.chunk_rows <= 0 or a.chunk_rows % 128: p.error('chunk rows must be a positive multiple of 128')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cache = a.output.with_suffix('.parts'); cache.mkdir(exist_ok=True)
    source_stat = a.gguf.stat()
    identity = {'source': str(a.gguf.resolve()), 'bytes': source_stat.st_size,
                'mtime_ns': source_stat.st_mtime_ns, 'encoder_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
                'converter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'only': a.only}
    identity_path = cache / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('conversion cache provenance differs; select a new output path')
    identity_path.write_text(json.dumps(identity, indent=2)+'\n')
    reader = gguf.GGUFReader(str(a.gguf))
    worker_log = (cache/'worker.log').open('a')
    worker = subprocess.Popen([str(a.worker.resolve()), str(a.source.resolve())], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=worker_log, text=True, bufsize=1)
    selector = selectors.DefaultSelector(); selector.register(worker.stdout, selectors.EVENT_READ)
    specs, paths, measurements = [], {}, []
    def add(name, tag, shape, path):
        specs.append(TensorSpec(name, tag, tuple(shape), path.stat().st_size)); paths[name] = path
    started = time.monotonic()
    try:
        for t in reader.tensors:
            if a.only and t.name != a.only: continue
            shape = tuple(int(x) for x in t.shape[::-1]); name = t.name
            path = cache / (name+'.bin'); receipt = cache/(name+'.json')
            quant = len(shape) in (2,3) and shape[-1] % 128 == shape[-2] % 128 == 0 and name != 'token_embd.weight'
            if receipt.exists():
                record = json.loads(receipt.read_text())
            else:
                stamp = time.monotonic(); sse = energy = seconds = 0.0
                if not quant:
                    dtype = np.float32 if len(shape) == 1 or 'gate_inp' in name else np.float16
                    with path.open('wb') as f:
                        if len(shape) < 2:
                            gguf.quants.dequantize(t.data, t.tensor_type).astype(dtype).tofile(f)
                        else:
                            for start in range(0, shape[0], a.chunk_rows):
                                gguf.quants.dequantize(t.data[start:start+a.chunk_rows], t.tensor_type).astype(dtype).tofile(f)
                    record = {'name':name, 'tag':0 if dtype == np.float32 else 1, 'shape':shape}
                else:
                    experts = shape[0] if len(shape) == 3 else 1; n,k = shape[-2:]
                    if k * a.chunk_rows > 8 * 1024 * 1024: raise ValueError('chunk exceeds worker bound')
                    packed = np.memmap(path, dtype='<u4', mode='w+', shape=(experts,k//16,n//16,32))
                    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
                    rng = np.random.default_rng(seed)
                    suh = rng.choice(np.array([-1,1],np.float16), size=(experts,k))
                    signs = rng.choice(np.array([-1,1],np.float16), size=(experts,n))
                    svh = np.empty((experts,n),np.float16)
                    for expert in range(experts):
                        data = t.data[expert] if len(shape)==3 else t.data
                        for start in range(0,n,a.chunk_rows):
                            end=min(start+a.chunk_rows,n)
                            weight = gguf.quants.dequantize(data[start:end], t.tensor_type).astype(np.float32)
                            scales = np.maximum(np.sqrt(np.mean(weight*weight,axis=1)),np.float32(2**-14)).astype(np.float16)
                            svh[expert,start:end] = scales * signs[expert,start:end]
                            transformed = weight.T / suh[expert,:,None] / svh[expert,None,start:end]
                            transformed = np.ascontiguousarray(fwht128(fwht128(transformed,0),1),dtype=np.float32)
                            input_path, output_path = cache/'chunk.f32',cache/'chunk.packed'
                            transformed.tofile(input_path)
                            request={'input':str(input_path.resolve()), 'output':str(output_path.resolve()),'k':k,'n':end-start}
                            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
                            if not selector.select(timeout=180): raise TimeoutError('Metal converter response timeout')
                            line=worker.stdout.readline()
                            if not line: raise RuntimeError('Metal converter exited; see '+str(cache/'worker.log'))
                            result=json.loads(line)
                            packed[expert,:,start//16:end//16,:] = np.fromfile(output_path,dtype='<u4').reshape(k//16,(end-start)//16,32)
                            sse += result['sse'];energy += result['energy'];seconds += result['gpu_seconds']
                    packed.flush(); del packed
                    suh.tofile(cache/(name+'.suh'));svh.tofile(cache/(name+'.svh'))
                    (cache/(name+'.exl3')).write_bytes(struct.pack('<8s6I',b'EXL3MUL1',1,4,k,n,128,0))
                    record={'name':name,'tag':17,'shape':shape,'normalized_sse':sse,'normalized_energy':energy,
                            'normalized_relative_rmse':float(np.sqrt(sse/energy)),'gpu_seconds':seconds}
                record['seconds']=time.monotonic()-stamp
                receipt.write_text(json.dumps(record,indent=2)+'\n')
            add(name,record['tag'],shape,path)
            if quant:
                prefix = shape[:-2]
                add(name+'.suh',1,(*prefix,shape[-1]),cache/(name+'.suh'))
                add(name+'.svh',1,(*prefix,shape[-2]),cache/(name+'.svh'))
                add(name+'.exl3',7,(32,),cache/(name+'.exl3'))
            measurements.append(record)
            print(json.dumps(record),flush=True)
        worker.stdin.close()
        if worker.wait(timeout=30): raise RuntimeError('Metal converter failed')
        if not specs: raise ValueError('no matching tensors')
        if a.extra_archive:
            with a.extra_archive.open('rb') as src:
                for spec in specs_from_archive(a.extra_archive):
                    if not spec.name.startswith('z_'): continue
                    if spec.dtype_tag not in (0,1) or spec.byte_length > 1024*1024: raise ValueError('extra constants must be bounded dense tensors')
                    src.seek(spec.offset); path=cache/(spec.name+'.bin');path.write_bytes(src.read(spec.byte_length))
                    add(spec.name,spec.dtype_tag,spec.shape,path)
        def chunks(name):
            with paths[name].open('rb') as f:
                while block := f.read(8*1024*1024): yield block
        summary=write_payloads(specs,a.output,chunks)
        result={'status':'passed','source':identity,'encoder':'mul1-4bit-lookahead2-three-sweeps',
                'seconds':time.monotonic()-started,'archive':str(a.output.resolve()),'archive_bytes':summary.file_bytes,
                'tensors':measurements,'metric':'RMSE in normalized rotated weight space; not output/model quality'}
        a.output.with_suffix('.conversion.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='tensors'}),flush=True)
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
        selector.close();worker_log.close()

if __name__=='__main__': main()
