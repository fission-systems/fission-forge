"""Execute freestanding i32 C-function modules with no imports/WASI/host callbacks.

Called as a bounded subprocess by functional.py. Never executes native candidates.
Each input gets a fresh instance, memory limit, and deterministic fuel budget.
"""
import argparse
import json
from pathlib import Path
import resource
import sys


def execute(module_bytes, inputs, *, fuel=100000, memory_bytes=1048576):
    import wasmtime
    config=wasmtime.Config();config.consume_fuel=True;config.max_wasm_stack=262144
    config.wasm_threads=False
    engine=wasmtime.Engine(config)
    module=wasmtime.Module(engine,module_bytes)
    if module.imports:
        return dict(status='sandbox_rejected',diagnostic='host imports are forbidden',cases=[])
    exports={export.name:export.type for export in module.exports}
    signature=exports.get('f')
    if not isinstance(signature,wasmtime.FuncType) or [str(t) for t in signature.params]!=['i32','i32'] or [str(t) for t in signature.results]!=['i32']:
        return dict(status='abi_rejected',diagnostic='expected exported f(i32,i32)->i32',cases=[])
    cases=[]
    for a,b in inputs:
        store=wasmtime.Store(engine)
        store.set_limits(memory_size=memory_bytes,table_elements=1024,instances=1,tables=1,memories=1)
        store.set_fuel(fuel)
        try:
            instance=wasmtime.Instance(store,module,[])
            f=instance.exports(store)['f']
            signed=lambda n:n if n<2**31 else n-2**32
            result=f(store,signed(a),signed(b))
            cases.append(dict(status='ok',value=result & 0xffffffff,fuel_remaining=store.get_fuel()))
        except wasmtime.Trap as error:
            status='fuel_exhausted' if error.trap_code==wasmtime.TrapCode.OUT_OF_FUEL else 'trap'
            cases.append(dict(status=status,diagnostic=str(error)[:1000]))
        except wasmtime.WasmtimeError as error:
            cases.append(dict(status='runtime_error',diagnostic=str(error)[:1000]))
    return dict(status='executed',cases=cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--fuel',type=int,required=True)
    parser.add_argument('--memory-bytes',type=int,required=True)
    args=parser.parse_args()
    # RLIMIT_AS is not reliable on macOS/JIT virtual reservations. Linear-memory,
    # stack and module-size bounds are enforced separately, CPU/file sizes here.
    resource.setrlimit(resource.RLIMIT_CPU,(5,6));resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    inputs=json.loads(args.inputs.read_text())
    if not (0<args.fuel<=10000000 and 131072<=args.memory_bytes<=16777216 and 0<len(inputs)<=1024):raise ValueError('invalid sandbox limits/input count')
    if any(len(row)!=2 or any(type(n) is not int or not 0<=n<2**32 for n in row) for row in inputs):raise ValueError('invalid uint32 inputs')
    data=args.module.read_bytes()
    if len(data)>1048576:raise ValueError('module size exceeds limit')
    print(json.dumps(execute(data,inputs,fuel=args.fuel,memory_bytes=args.memory_bytes)))


if __name__=='__main__':main()
