import os
import subprocess

# 指定Solver子目录的路径
script_directory = os.path.dirname(os.path.abspath(__file__))
metallib_path = script_directory + "/../resources/metal_libs"
solver_dirs = [script_directory + "/Dynamics", 
               script_directory + "/LBVH",  
               script_directory + "/Accelerator/Vivace",
               script_directory + "/Shaders"]
utils_dirs = [script_directory + "/SharedDefine", 
              script_directory + "/Dynamics/shared", 
              script_directory + "/LBVH/shared",
              script_directory + "/Accelerator/Vivace/shared"]
# glm_dir = "/Users/huohuo/Downloads/glm-master/glm"

# binary_dir = script_directory + "/build"
# print(script_directory)
# print(solver_dir)

# 检查 metallib_path 是否存在
if not os.path.exists(metallib_path):
    os.makedirs(metallib_path)


# 遍历Solver子目录下的.metal文件
for solver_dir in solver_dirs:
    for root, dirs, files in os.walk(solver_dir):
        for file in files:
            if file.endswith(".metal"):
                metal_file = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                metallib_file = os.path.join(metallib_path, f"{base_name}.metallib")

                # 构建metal编译命令
                include_dirs = " ".join([f"-I {dir}" for dir in utils_dirs])
                command = f"xcrun -sdk macosx metal {include_dirs} -Os {metal_file} -o {metallib_file} "
                # command = f"xcrun -sdk macosx metal -I {utils_dir} -I {glm_dir} -Os {metal_file} -o {metallib_file} "

                # 执行编译命令
                try:
                    subprocess.run(command, shell=True, check=True)
                    print(f"编译成功: {file} -> {base_name}.metallib")
                except subprocess.CalledProcessError as e:
                    print(f"编译失败: {file}\n错误信息: {e.stderr}")
                except Exception as e:
                    print(f"执行命令时出现错误: {e}")

print("ok")
# command = f"cmake -G \"Xcode\" .."
# if not os.path.exists(binary_dir):
#     os.makedirs(binary_dir)
#     print(f"文件夹 '{binary_dir}' 已创建")
# else:
#     print(f"文件夹 '{binary_dir}' 已存在")
# subprocess.run(command, cwd=binary_dir, shell=True, check=True)