def visualize_breath(data):
    _id = data["breath_id"].iloc[0]
    r = data["R"].iloc[0]
    c = data["C"].iloc[0]
    print("total rows {}".format(len(data)))
    print("R {}, C {}".format(r, c))
    data.plot(
        x="time_step", y=["u_in", "u_out", "pressure"], title="breath id {}".format(_id)
    )
