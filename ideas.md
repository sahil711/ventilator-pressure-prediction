1. treat u_out as a cat variable and use embedding of higher dim
2. treat R and C as cat variable and use embedding of higher dim
3. for feature creation like lead/lag use conv1d as a frontend and then pass to the differnt backbones
4. loss propogation on whole seq vs loss propogation using only u_out=0
5. truncate the seq to len 40, will reduce compute time too (as u_out=0 is at max present till 35)

blocks:
1. resnet like blocks (with and w/o SE)
2. conformers blocks
3. wavenet blocks

