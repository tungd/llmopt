#!/usr/bin/env python3
"""Sequential full-model EXL3 / source-GGUF performance receipt.

llama-bench uses its own synthetic inputs; these are throughput workloads, not
matched-token quality tests. Retain TTFT and all inter-token gaps as well as the
existing llmopt steady metric (which excludes its first inter-token gap).
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine',type=Path,required=True);p.add_argument('--gguf',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--prompt',type=int,default=64)
p.add_argument('--generate',type=int,default=64);p.add_argument('--llama-bench',default='/opt/local/bin/llama-bench')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
result={'status':'running','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'engine':str(a.engine.resolve()),'gguf':str(a.gguf.resolve()),'prompt_tokens':a.prompt,'generated_tokens':a.generate,
        'comparison_scope':__doc__,'steps':[]}
def save(): (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
def run(name,command):
 prefix=a.output/name
 subprocess.run([sys.executable,str(ROOT/'bench/run_serial_job.py'),'--output',str(prefix),'--timeout','1800','--',*command],check=True)
 receipt=json.loads(prefix.with_suffix('.json').read_text());result['steps'].append(receipt);save()
 return prefix.with_suffix('.log').read_text()
def metrics(text):
 d={}
 for line in text.splitlines():
  if ':' not in line:continue
  k,v=line.split(':',1)
  try:d[k]=float(v)
  except ValueError:d[k]=v.strip()
 return d
save()
try:
 bench=str(ROOT/'_build/bin/llmopt-bench-tokens')
 result['llmopt_prefill']=metrics(run('llmopt-prefill',[bench,'--model-dir',str(a.engine.resolve()),'--prompt-token-ids',','.join(map(str,range(1,a.prompt+1))),'--top-k-logits','5','--warmup']))
 result['llmopt_decode']=metrics(run('llmopt-decode',[bench,'--model-dir',str(a.engine.resolve()),'--prompt-token-ids','1,2,3,4','--max-new-tokens',str(a.generate),'--warmup']))
 raw=run('llama',[a.llama_bench,'-m',str(a.gguf.resolve()),'-ngl','99','-r','3','-p',str(a.prompt),'-n',str(a.generate),'-o','json'])
 # The controller combines stderr and stdout; locate llama-bench's JSON array.
 decoder=json.JSONDecoder();data=None
 for i,c in enumerate(raw):
  if c=='[':
   try:
    candidate,_=decoder.raw_decode(raw[i:])
    if isinstance(candidate,list) and candidate and isinstance(candidate[0],dict) and 'avg_ts' in candidate[0]:data=candidate;break
   except json.JSONDecodeError:pass
 if data is None:raise ValueError('llama-bench JSON missing; retained full log')
 result['llama']=data
 pp=next(x['avg_ts'] for x in data if x['n_prompt']==a.prompt and x['n_gen']==0)
 tg=next(x['avg_ts'] for x in data if x['n_prompt']==0 and x['n_gen']==a.generate)
 prefill=a.prompt/result['llmopt_prefill']['probe-seconds']
 decode=result['llmopt_decode'];full=(decode['completion-token-count']-1)/(decode['elapsed-seconds']-decode['ttft-seconds'])
 result['comparison']={'llmopt_prefill_tps':prefill,'llama_prefill_tps':pp,'prefill_ratio':prefill/pp,
  'llmopt_steady_decode_tps':decode['steady-tokens-per-second'],'llmopt_all_decode_gaps_tps':full,'llama_decode_tps':tg,
  'steady_decode_ratio':decode['steady-tokens-per-second']/tg,'all_decode_gaps_ratio':full/tg}
 result['status']='passed';save();print(json.dumps(result['comparison'],indent=2))
except BaseException as e:
 result['status']='failed';result['error']=str(e);save();raise
