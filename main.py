from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import draw

# Initialize weights and biases
images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))
leave = 0
learn_rate = 0.04
nr_correct_0 = 0
nr_correct_1 = 0
nr_correct_2 = 0
nr_correct_3 = 0
nr_correct_4 = 0
nr_correct_5 = 0
nr_correct_6 = 0
nr_correct_7 = 0
nr_correct_8 = 0
nr_correct_9 = 0
nr_correct_all = 0
nr_correct = [nr_correct_0, nr_correct_1, nr_correct_2, nr_correct_3, nr_correct_4, nr_correct_5, nr_correct_6,
              nr_correct_7, nr_correct_8, nr_correct_9, nr_correct_all]

while True:
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input into hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden into output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct[int(np.argmax(o))] += int(np.argmax(o) == np.argmax(l))
        nr_correct[10] += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output into hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden into input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch

    print(f"Acc: {round((nr_correct[10] / images.shape[0]) * 100, 2)}%")
    for i in range(len(nr_correct) - 1):
        if round((nr_correct[i] / images.shape[0]) * 100, 2) > 8:
            leave += 1
            print(str(i) + ": passes with a " + str(round((nr_correct[i] / images.shape[0]) * 100, 2)) + "% accuracy")
            nr_correct[i] = 0
        else:
            print(f"Acc {i}: {round((nr_correct[i] / images.shape[0]) * 100, 2)}%")
            nr_correct[i] = 0
    if leave >= 10:
        break
    else:
        leave = 0
        nr_correct[10] = 0

# Show results
while True:
    draw.main()
    img = plt.imread("Playerdrawinggray.jpg")

    img.shape += (1,)
    # Forward propagation input into hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden into output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    # Backpropagation output into hidden (cost function derivative)
    l = np.zeros((10, 1))
    print("Is it: " + str(o.argmax()))
