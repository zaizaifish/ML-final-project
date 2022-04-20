import matplotlib.pyplot as plt


resnet = [9.906258374452591, 10.046567767858505, 11.274479120969772, 10.129723370075226, 8.681421309709549, 10.335718750953674, 9.933183401823044, 8.402258425951004, 8.530503571033478, 9.002842009067535]
cnn = [13.453681528568268, 11.4856615960598, 10.705304741859436, 10.939388483762741, 9.69030675292015, 9.456131994724274, 9.155956268310547, 9.18941581249237, 8.722096890211105, 8.143218070268631]
iterations = [i + 1 for i in range(len(cnn))]
plt.plot(iterations, cnn, 'xb-', label = "cnn", color = "black")
plt.plot(iterations, resnet, 'ob-', label = "resnet", color = "blue")
leg = plt.legend(loc = 'best')
plt.title("Loss against iteration times")
plt.xlabel("iteration")
plt.ylabel("Crossentropy Loss")
plt.savefig("Loss against iteration times")
