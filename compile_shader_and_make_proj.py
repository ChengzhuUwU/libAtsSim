import os
import subprocess

# 指定Solver子目录的路径
script_directory = os.path.dirname(os.path.abspath(__file__))
metallib_path = script_directory + "/resources/metal_libs"
solver_dirs = [script_directory + "/examples", 
               ]
utils_dirs = [script_directory + "/src/SharedDefine", 
              script_directory + "/src/Solver"]


if not os.path.exists(metallib_path):
    os.makedirs(metallib_path)


for solver_dir in solver_dirs:
    for root, dirs, files in os.walk(solver_dir):
        for file in files:
            if file.endswith(".metal"):
                metal_file = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                metallib_file = os.path.join(metallib_path, f"{base_name}.metallib")

                # 
                include_dirs = " ".join([f"-I {dir}" for dir in utils_dirs])
                command = f"xcrun -sdk macosx metal {include_dirs} -Os {metal_file} -o {metallib_file} "
                # command = f"xcrun -sdk macosx metal -I {utils_dir} -I {glm_dir} -Os {metal_file} -o {metallib_file} "

                # 
                try:
                    subprocess.run(command, shell=True, check=True)
                    print(f"Compile succ: {file} -> {base_name}.metallib")
                except subprocess.CalledProcessError as e:
                    print(f"Compile failed: {file}")
                    print(f"Error: {e.stderr}")
                except Exception as e:
                    print(f"Error in running compiling command: {e}")

print("Compile done")