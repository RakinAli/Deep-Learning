            steps += 1
            if steps % 20 == 0:
                acc = compute_accuracy(data_train, labels_train, weights, bias)
                print("Step: ", steps, " Accuracy: ", acc)
