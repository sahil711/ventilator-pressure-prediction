1. treat u_out as a cat variable and use embedding of higher dim
2. treat R and C as cat variable and use embedding of higher dim
3. for feature creation like lead/lag use conv1d as a frontend and then pass to the differnt backbones

frontend:
    1. wavenet
    2. gru/lstm

head:
    1. gru/lstm
    2. wavenet

