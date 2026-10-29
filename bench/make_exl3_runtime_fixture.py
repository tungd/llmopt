"""Build ordinary and routed-MoE reference outputs with the pinned upstream decoder."""
import argparse, importlib.util, json, struct, subprocess
from pathlib import Path
import numpy as np
import torch
from llmopt_backend.tensor_archive import TensorSpec,write_payloads
from make_exl3_fixture import REVISION
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
refpath=a.upstream/'tests/testlib/trellis.py'
assert refpath.read_bytes()==subprocess.check_output(['git','-C',str(a.upstream),'show',REVISION+':tests/testlib/trellis.py'])
s=importlib.util.spec_from_file_location('reference',refpath);ref=importlib.util.module_from_spec(s);s.loader.exec_module(ref)
torch.set_num_threads(1);rng=np.random.default_rng(712904);h=ref.sylvester(128).numpy()/np.sqrt(128)
payloads={}
def add(name,tag,value): payloads[name]=(tag,value.shape,value.tobytes())
def weights(name,experts,k,n):
 ps=[];decoded=[]
 su=rng.choice([-1,1],size=(experts,k)).astype(np.float16)
 sv=(rng.uniform(.03,.07,size=(experts,n))*rng.choice([-1,1],size=(experts,n))).astype(np.float16)
 for _ in range(experts):
  bits=ref.random_packed(k//16*(n//16),4,rng).reshape(k//16,n//16,64)
  ps.append(bits.tobytes());decoded.append(ref.dequant_rotated(bits,4,'mul1'))
 shape=(experts,n,k) if experts>1 else(n,k)
 payloads[name]=(17,shape,b''.join(ps));add(name+'.suh',1,su if experts>1 else su[0]);add(name+'.svh',1,sv if experts>1 else sv[0])
 payloads[name+'.exl3']=(7,(32,),struct.pack('<8s6I',b'EXL3MUL1',1,4,k,n,128,0))
 def project(x,e=0):
  xx=((x.astype(np.float64)*su[e]).reshape(-1,128)@h).reshape(-1,k).astype(np.float16)
  y=(xx.astype(np.float64)@decoded[e].astype(np.float64)).astype(np.float32)
  return (((y.astype(np.float64).reshape(-1,128)@h).reshape(-1,n))*sv[e]).astype(np.float16)
 return project
linear=weights('linear',1,128,256);gate=weights('gate',3,128,256);up=weights('up',3,128,256);down=weights('down',3,256,128)
x=rng.uniform(-.4,.4,size=(64,128)).astype(np.float16);router=rng.uniform(-1,1,size=(64,3)).astype(np.float32)
router[:32,2] = -10  # Empty expert for m=17; mixed/partial tiles for m=64.
add('input',1,x);add('router',0,router);add('expected.linear',1,linear(x))
y=np.zeros((64,128),np.float32)
for i in range(64):
 indices=np.argsort(router[i])[::-1][:2];prob=np.exp(router[i,indices]-max(router[i,indices]));prob=(prob/prob.sum()).astype(np.float16)
 for j,e in enumerate(indices):
  g=gate(x[i:i+1],e).astype(np.float32);u=up(x[i:i+1],e).astype(np.float32)
  z=(g/(1+np.exp(-g))*u).astype(np.float16)
  y[i]+=down(z,e)[0].astype(np.float32)*np.float32(prob[j])
add('expected.moe',1,y.astype(np.float16))
write_payloads([TensorSpec(name,tag,shape,len(data)) for name,(tag,shape,data) in payloads.items()],a.output,lambda name:(payloads[name][2],))
print(json.dumps({'status':'passed','archive':str(a.output),'oracle_revision':REVISION,'rows':64,'experts':3,'top_k':2}))
