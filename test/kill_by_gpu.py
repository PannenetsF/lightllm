import os
import argparse
import gpustat


def kill_by_num(ids):
    data = gpustat.core.new_query().jsonify()["gpus"]
    selected_gpus = [d for d in data if d["index"] in ids]
    for gpu in selected_gpus:
        gid = gpu["index"]
        for proc in gpu["processes"]:
            pid = proc["pid"]
            cmd = " ".join(proc["full_command"])
            os.system(f"kill -9 {pid}")
            print(f"killed {pid} on GPU {gid} with command {cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", type=str, default="")
    args = parser.parse_args()
    ids = [int(i) for i in args.ids.split(",")]
    kill_by_num(ids)
    print("done")
