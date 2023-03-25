def get_num_pos_neg(data):
    labels = [d.y.item() for d in data]
    pos = labels.count(1)
    neg = labels.count(0)

    return pos, neg

