## Distributed LaNAS
• <b>accurate evaluations, suit for chasing SoTA results</b>

• <b>you need a lot of GPUs</b>

This provides a simple distributed framework for training using LaNAS, with which we achieve SoTA results with 500 GPUs. The distributed LaNAS trains every sampled network from scratch, and I believe techniques such as early prediction will be a very nice improvement to
the current implementation. Because sending network configurations is fairly cheap, we implemented a simple client-server system to parallelize the distributed search. This figure depicts the general idea.

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/distributed_lanas_architecture.png?raw=true' width="500">
</p>

## Starting the server

We uniformly sampled a few million networks from the NASNet search space, and pre-built search space in the file of "search_space". The server loads the file, and search the networks within the file. Feel free to change this to a random generator and merge with this branch.

Here are the steps to start:
1. go to server folder, unzip search_space.zip.
2. ifconfig get your ip address
3. you need change the line 212 in MCTS.py
```
address = ('XXX.XX.XX.XXX', 8000), # replace XX to your ip address, and change to different ports if 8000 does not work.
```
4. To start the server, ``` python MCTS.py & ```.

## Starting the clients
Each client folder corresponds to a GPU; you can create as many clients folder as you want, simply copy and paste.

Once the server starts running, here is what you need to start clients.
1. go to client folder, open client.py
2. change line 20, line 71, line 109 to <b>the server's ip address</b>.
3. set to a unused GPU
4. python client.py

If you have 500 GPUs, create 500 folders, and repeat the above process 500x. ;)

## Collecting the results
We write a script collect_results.py to collect all the results in client folders. Once it creates total_trace.json (we also uploaded the total trace collected from our experiments), you can read the results by ``` python read_results.py```, and the results are ranked backward, i.e. the last row is the best.

Here is the snapshot of best architectures found in our distribtued search.
<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/distributed_search_results.png?raw=true' width="600">
</p>

The last column is the test accuracy after training each networks for 200 epochs. We assume the best network is the one with the best test accuracy. 

## Training the top model
You can train the best "searched" network using the training pipeline <a href="../LaNet/CIFAR10">here</a>. 

## Fault Tolerance
Fault tolerance is very important if you will use hundreds of GPUs. We have already taken care of it in the current implementation.

On the server side, it will dump the pickled current state at every search iteration in the file named "mcts_agent". You can resume the searching with that state. The MCTS.py will find mcts_agent in the current folder. If your server got preempted, simply python MCTS.py again.

On the client side, it will dump the training state, and resume the training if a job was preempted in the middle of training. To restart a client, python client.py. That's it. ;)
