import matplotlib.pyplot as plt
from utils import Perceptron


def input_learning_points(first_set_size, second_set_size):
    plt.ion()
    plt.title(f"Mark {first_set_size} points of first set and then {second_set_size} points of second set")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    first_set = plt.ginput(first_set_size, timeout=-1)
    plt.scatter([point[0] for point in first_set], [point[1] for point in first_set], color="red", marker="x")
    second_set = plt.ginput(second_set_size, timeout=-1)
    plt.scatter([point[0] for point in second_set], [point[1] for point in second_set], color="blue", marker=".")
    return first_set, second_set


first_set_size = 5
second_set_size = 6
first_set, second_set = input_learning_points(first_set_size, second_set_size)

points = first_set + second_set
target_values = [1]*first_set_size + [0]*second_set_size

perceptron = Perceptron()
perceptron.train(points, target_values)

linear_function = lambda x: (-perceptron.w1*x - perceptron.wb)/perceptron.w2
y1 = linear_function(0)
y2 = linear_function(10)

plt.title("Click to select prediction point, close window to finish")
plt.plot([0, 10], [y1, y2])

while True:
    try:
        point = plt.ginput(1, timeout=-1)[0]
        prediction = perceptron.predict(point)
        if prediction == 1:
            plt.scatter(point[0], point[1], color="red", marker="+")
        else:
            plt.scatter(point[0], point[1], color="blue", marker="+")
        print(f"point: {point} set: {'first (red)' if prediction == 1 else 'second (blue)'}")
    except:
        break
