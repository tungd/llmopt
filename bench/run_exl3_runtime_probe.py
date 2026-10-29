"""Build and execute the native EXL3 archive/schedule/ordinary/MoE integration test."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
def run(name,cmd):
 subprocess.run([sys.executable,str(ROOT/'bench/run_serial_job.py'),'--output',str(a.output/name),'--timeout','600','--',*cmd],check=True)
run('build',['ninja','-f','ninja.build','-j1','_build/bin/llmopt-exl3-runtime-test'])
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'python')+':'+str(ROOT/'bench')
subprocess.run([str(ROOT/'python/.venv/bin/python'),'bench/make_exl3_runtime_fixture.py','--upstream',str(a.upstream),'--output',str(a.output/'weights.llmopt')],env=env,check=True,timeout=120)
run('device',[str(ROOT/'_build/bin/llmopt-exl3-runtime-test'),str(a.output)])
