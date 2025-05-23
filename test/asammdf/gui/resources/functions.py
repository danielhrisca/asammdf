import numpy as np


def Function1(t=0):
    sequence = range(10)
    filtered = list(filter(lambda i: i % 2, sequence))
    if filtered:
        for _ in filtered:
            if not _:
                continue
            try:
                int(_)
            except ValueError:
                print(str(_))
            break
        else:
            return 1
        return 2
    else:
        return 0


def Function2(t=0):
    sequence = range(10)
    filtered = filter(lambda i: i % 2, sequence)
    if len(filtered):
        for _ in filtered:
            if not _:
                continue
            try:
                int(_)
            except ValueError:
                print(str(_))
            break
        else:
            return 1
    else:
        return 0


def UnresolvedVariable(t=0):
    a, b = 0, 1
    if a:
        c = 0
    return c


def WrongDefinition():
    return 1


def gray2dec(position_sensor_value=0, t=0):
    for shift in (8, 4, 2, 1):
        position_sensor_value = position_sensor_value ^ (position_sensor_value >> shift)

    return position_sensor_value


def maximum(channel1=0, channel2=-1, channel3=0, t=0):
    return max(channel1, channel2, channel3)


def rpm_to_rad_per_second(speed=0, t=0):
    return 2 * np.pi * speed / 60
