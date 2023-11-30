from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import draw
import keyboard
import pandas as pd
import os
# Initialize weights and biases

images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (1000, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 1000))
b_i_h = np.zeros((1000, 1))
b_h_o = np.zeros((10, 1))
leave = 0
learn_rate = 0.01
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
save_weights = True
run = True
while True:
    training = input("Do you want to train the network? (y/n)\n")
    if training in ["y", "Y", "yes", "Yes", "YES"]:
        break
    elif training in ["n", "N", "no", "No", "NO"]:
        run = False
        save_weights = False
        break
    else:
        print("Please enter a valid input such as (y/n)")
while run:
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
        if keyboard.is_pressed('`'):
            break
    # Show accuracy for this epoch

    print(f"Acc: {round((nr_correct[10] / images.shape[0]) * 100, 2)}%")
    for i in range(len(nr_correct) - 1):
        if round((nr_correct[i] / images.shape[0]) * 100, 2) >= 9.9:
            leave += 1
            print(str(i) + ": passes with a " + str(round((nr_correct[i] / images.shape[0]) * 100, 2)) + "% accuracy")
            nr_correct[i] = 0
        else:
            print(f"Acc {i}: {round((nr_correct[i] / images.shape[0]) * 100, 2)}%")
            nr_correct[i] = 0
    if leave >= 10 or keyboard.is_pressed('`'):
        break
    else:
        leave = 0
        nr_correct[10] = 0

while save_weights:
    move_on = input("Do you want to save the weights? (y/n)\n")
    # Check if the DataFrame is empty
    if move_on in ["y", "Y", "yes", "Yes", "YES"]:
        df = pd.DataFrame(w_i_h)

        # Specify the full path for the CSV file
        file_path = r'C:\Code\Python\NumberIdentification\w_i_h.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        # Save the DataFrame to a CSV file, creating the file if it doesn't exist
        df.to_csv(file_path, index=False, header=False, mode='x')

        df = pd.DataFrame(w_h_o)

        # Specify the full path for the CSV file
        file_path = r'C:\Code\Python\NumberIdentification\w_h_o.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        # Save the DataFrame to a CSV file, creating the file if it doesn't exist
        df.to_csv(file_path, index=False, header=False, mode='x')

        df = pd.DataFrame(b_i_h)

        # Specify the full path for the CSV file
        file_path = r'C:\Code\Python\NumberIdentification\b_i_h.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        # Save the DataFrame to a CSV file, creating the file if it doesn't exist
        df.to_csv(file_path, index=False, header=False, mode='x')

        df = pd.DataFrame(b_h_o)

        # Specify the full path for the CSV file
        file_path = r'C:\Code\Python\NumberIdentification\b_h_o.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        # Save the DataFrame to a CSV file, creating the file if it doesn't exist
        df.to_csv(file_path, index=False, header=False, mode='x')
        break
    elif move_on in ["n", "N", "no", "No", "NO"]:
        # Load data from the CSV file into a DataFrame
        file_path = r'C:\Code\Python\NumberIdentification\w_h_o.csv'
        w_h_o = pd.read_csv(file_path, header=None).values
        file_path = r'C:\Code\Python\NumberIdentification\w_i_h.csv'
        w_i_h = pd.read_csv(file_path, header=None).values
        file_path = r'C:\Code\Python\NumberIdentification\b_h_o.csv'
        b_h_o = pd.read_csv(file_path, header=None).values
        file_path = r'C:\Code\Python\NumberIdentification\b_i_h.csv'
        b_i_h = pd.read_csv(file_path, header=None).values
        break
    else:
        print("Please enter a valid input")

# Show results
while True:
    draw.main()
    img = plt.imread("CenteredDrawinggray.jpg")

    img.shape += (1,)
    # Forward propagation input into hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden into output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))
    print("Is it: " + str(o.argmax()))
    for i in range(len(o)):
        print(f"{i}: {round(o[i][0] * 100, 2)}%")
    if keyboard.is_pressed('`'):
        break
