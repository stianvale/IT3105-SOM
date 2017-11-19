import som
import somMnist

def run():

	userinpt = ""


	while userinpt != "exit":
		userinpt = input("Please choose either 'tsp' or 'mnist': ")

		if userinpt == "tsp":
			fileNumber = input("Please choose the tsp-problem to run: ")
			som.main(fileNumber)
		elif userinpt =="mnist":
			somMnist.main()
		else:
			print("Invalid input, please try again: ")

run()
