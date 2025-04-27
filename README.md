# HeteroScheduler

The source code of **SIGGRAPH 2025** paper **Auto Task Scheduling for Cloth and Deformable Simulation on Heterogeneous Environments** (Chengzhu He, Zhendong Wang, Zhaorui Meng, Junfeng Yao, Shihui Guo, Huamin Wang).

We provided several examples to show our scheduler and asynchronous iteration progress.

## Example 1: Simplist HEFT case

This example shows how HEFT algorithm makes scheduling.

Our HEFT implementation is based on [python-version heft](https://github.com/mackncheesiest/heft), which is the source code of paper on **IEEE Transactions on Parallel and Distributed Systems 2022**: "Performant, Multi-Objective Scheduling of Highly Interleaved Task Graphs on Heterogeneous System on Chip Devices" (J. Mack, S. E. Arda, U. Y. Ogras and A. Akoglu).

## Example 2: Asychronous iteration on VBD

This example shows the difference between the original iteration pipeline and our asynchronous iteration pileline. Considering we have allocated iteration tasks (different colors) into 2 devices, then how do we make data transfering? 

This example only considering the simplist case: costs of tasks is a constant value $t_c$ and the comminication delay is exactly the same value $t_c$, then we make make the data transfering as follows:

![Example 2 case](example_2.png)

We use mass-spring streching and quadratic bending model, this can also extended to non-linear energy model. VBD (Vertex Block Descent) is a SIGGRAPH 2024 paper.

## Example 3: Asychronous iteration with CPU-GPU implementation

This example shows how do we use our heterogenous framework in a simulation application. After we register the implementation and specify DAG, the our scheduler will automatically make scheuling including calculating the communication matrix, allocating the tasks into devices, specifying the data tranfers.

We use Metal-shading-language for GPU implementation, so this example is only supported on MacOS.

## Dependencies

The library itself depends only on glm and TBB. For windows users, TBB installed by vcpkg might be hard to debug, so we use the source code to compile. Example 3 can only run on MacOS due to our Metal based GPU implementation.