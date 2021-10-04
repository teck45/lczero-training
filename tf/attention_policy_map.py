import numpy as np


move = np.arange(1, 8)

diag = np.array([
    move    + move*8,
    move    - move*8,
    move*-1 - move*8,
    move*-1 + move*8
])

orthog = np.array([
    move,
    move*-8,
    move*-1,
    move*8
])

knight = np.array([
    [2 + 1*8],
    [2 - 1*8],
    [1 - 2*8],
    [-1 - 2*8],
    [-2 - 1*8],
    [-2 + 1*8],
    [-1 + 2*8],
    [1 + 2*8]
])

promos = np.array([2*8, 3*8, 4*8])
pawn_promotion = np.array([
    -1 + promos,
    0 + promos,
    1 + promos
])


def make_map():
    """theoretically possible put-down squares (numpy array) for each pick-up square (list element).
    squares are [0, 1, ..., 63] for [a1, b1, ..., h8]. squares after 63 are promotion squares (1st draft concept).
    each successive "row" beyond 63 (ie. 64:72, 72:80, 80:88) are for over-promotions to queen, rook, and bishop;
    respectively. a pawn traverse to row 56:64 signifies a "default" promotion to a knight."""
    traversable = []
    for i in range(8):
        for j in range(8):
            sq = (8*i + j)
            traversable.append(
                sq +
                np.sort(
                    np.int32(
                        np.concatenate((
                            orthog[0][:7-j], orthog[2][:j], orthog[1][:i], orthog[3][:7-i],
                            diag[0][:np.min((7-i, 7-j))], diag[3][:np.min((7-i, j))],
                            diag[1][:np.min((i, 7-j))], diag[2][:np.min((i, j))],
                            knight[0] if i < 7 and j < 6 else [], knight[1] if i > 0 and j < 6 else [],
                            knight[2] if i > 1 and j < 7 else [], knight[3] if i > 1 and j > 0 else [],
                            knight[4] if i > 0 and j > 1 else [], knight[5] if i < 7 and j > 1 else [],
                            knight[6] if i < 6 and j > 0 else [], knight[7] if i < 6 and j < 7 else [],
                            pawn_promotion[0] if i == 6 and j > 0 else [],  # for promotion keys
                            pawn_promotion[1] if i == 6           else [],  # for promotion keys
                            pawn_promotion[2] if i == 6 and j < 7 else [],  # for promotion keys
                        ))
                    )
                )
            )
    # print(traversable)
    # print(traversable[-16:-8])
    # print(sum([a.__len__() for a in traversable]))

    # for static promotion logits:
    # z = np.zeros((64*64, 1858-66), dtype=np.int32)

    # for promotion keys:
    z = np.zeros((64*64+8*24, 1858), dtype=np.int32)

    # first loop for standard moves (for i in 0:1858, stride by 1)
    i = 0
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index < 64:
                z[putdown_index + (64*pickup_index), i] = 1
                i += 1

    """2nd draft promotion keys -- 2nd loop for promotions (for i in 1792:1858, stride by ls[j])
    this strange stride pattern is because the order of attention policy promotion logits naturally is
        a7a8q, a7b8q, a7a8r, a7b8r, ... , h7h8b
    but the order of lc0 policy index for promotions is
        a7a8q, a7a8r, a7a8b, a7b8q, ... , h7h8b
    """
    j = 0
    j1 = np.array([3, -2, 3, -2, 3])
    j2 = np.array([3, 3, -5, 3, 3, -5, 3, 3, 1])
    ls = np.append(j1, 1)
    for k in range(6):
        ls = np.append(ls, j2)
    ls = np.append(ls, j1)
    ls = np.append(ls, 0)
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index >= 64:
                pickup_file = pickup_index % 8
                z[4096 + pickup_file*24 + putdown_index-64, i] = 1
                i += ls[j]
                j += 1

    return z


# z = make_map()
# print(sum(sum(z)))
# print(np.shape(z))

# promotion concept 1
# 1792 = a7a8q  48*88 + 64 |#1
# 1793 = a7a8r  48*88 + 72 |#3
# 1794 = a7a8b  48*88 + 80 | 5
# 1795 = a7b8q  48*88 + 65 |#2
# 1796 = a7b8r  48*88 + 73 | 4
# 1797 = a7b8b  48*88 + 81 | 6
#
# 1798 = b7a8q  49*88 + 64 | 7
# 1799 = b7a8r  49*88 + 72 | 10
# 1800 = b7a8b  49*88 + 80 | 13
# 1801 = b7b8q  49*88 + 65 | 8
# 1802 = b7b8r  49*88 + 73 | 11
# 1803 = b7b8b  49*88 + 81 | 14
# 1804 = b7c8q  49*88 + 66 | 9
# 1805 = b7c8r  49*88 + 74 | 12
# 1806 = b7c8b  49*88 + 82 | 15
#
# ...

# promotion concept 2
# 1792 = a7a8q  1792 + file*24 + pp_row*8 + pp_file |#1
# 1793 = a7a8r  1792 +    0*24 +      1*8 +       0 |#3
# 1794 = a7a8b  1792 +    0*24 +      2*8 +       0 | 5
# 1795 = a7b8q  1792 +    0*24 +      0*8 +       1 |#2
# 1796 = a7b8r  1792 +    0*24 +      1*8 +       1 | 4
# 1797 = a7b8b  1792 +    0*24 +      2*8 +       1 | 6
#
# 1798 = b7a8q  1792 +    1*24 +      0*8 +       0 | 7
# 1799 = b7a8r  1792 +    1*24 +      1*8 +       0 | 10
# 1800 = b7a8b  1792 +    1*24 +      2*8 +       0 | 13
# 1801 = b7b8q  1792 +    1*24 +      0*8 +       1 | 8
# 1802 = b7b8r  1792 +    1*24 +      1*8 +       1 | 11
# 1803 = b7b8b  1792 +    1*24 +      2*8 +       1 | 14
# 1804 = b7c8q  1792 +    1*24 +      0*8 +       2 | 9
# 1805 = b7c8r  1792 +    1*24 +      1*8 +       2 | 12
# 1806 = b7c8b  1792 +    1*24 +      2*8 +       2 | 15
#
# 1807 = c7b8q  1792 +    2*24 +      0*8 +       1 | 16
# 1808 = c7b8r  1792 +    2*24 +      1*8 +       1 | 19
# 1809 = c7b8b  1792 +    2*24 +      2*8 +       1 | 22
# 1810 = c7c8q  1792 +    2*24 +      0*8 +       2 | 17
# 1811 = c7c8r  1792 +    2*24 +      1*8 +       2 | 20
# 1812 = c7c8b  1792 +    2*24 +      2*8 +       2 | 23
# 1813 = c7d8q  1792 +    2*24 +      0*8 +       3 | 18
# 1814 = c7d8r  1792 +    2*24 +      1*8 +       3 | 21
# 1815 = c7d8b  1792 +    2*24 +      2*8 +       3 | 24
#
# ...
