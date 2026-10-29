"""Run one bounded heavy stage, retaining its log and completion receipt."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--timeout',type=float,default=1800)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args(); command=a.command[1:] if a.command[:1]==['--'] else a.command
def active_workloads():
 active=[]
 for line in subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True).splitlines():
  f=line.strip().split(None,2)
  if len(f)<3 or int(f[0])==os.getpid(): continue
  exe=Path(f[1]).name
  if exe.startswith(('llmopt-','metal-exl3-','llama-bench','llama-server')) or exe in ('ninja','ocamlopt','swiftc','metal','metallib') or (exe.startswith('python') and any(Path(x).name in ('build_mellum21_engine.py','convert_gguf_exl3.py') for x in f[2].split())): active.append(line.strip())
 return active
active=active_workloads()
if active:
 print('Waiting for existing workload to finish: '+ '\n'.join(active),flush=True)
 deadline=time.monotonic()+600
 while active and time.monotonic()<deadline:
  time.sleep(2);active=active_workloads()
 if active:raise SystemExit('Not started: another workload remains active after the monitoring deadline')
a.output.parent.mkdir(parents=True,exist_ok=True)
log=a.output.with_suffix('.log');receipt=a.output.with_suffix('.json');t=time.monotonic()
r={'job':a.output.name,'controller_pid':os.getpid(),'command':command,'log':str(log),'status':'running'}
with log.open('w') as stream:
 child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 r['pid']=child.pid;receipt.write_text(json.dumps(r,indent=2)+'\n')
 try: code=child.wait(timeout=a.timeout);r['status']='passed' if code==0 else 'failed'
 except subprocess.TimeoutExpired:
  r['status']='timed_out'
  for sig, grace in ((signal.SIGTERM,10),(signal.SIGKILL,5)):
   try: os.killpg(child.pid,sig)
   except ProcessLookupError: pass
   except PermissionError as error: r['cleanup_error']=str(error)
   try:
    code=child.wait(timeout=grace);r['status']='timed_out_and_reaped';break
   except subprocess.TimeoutExpired: pass
  else:
   code=child.poll();r['status']='timed_out_outcome_unknown' if code is None else 'timed_out_and_reaped'
r.update(exit_code=code,seconds=time.monotonic()-t);receipt.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r));raise SystemExit(0 if r['status']=='passed' else 1)
