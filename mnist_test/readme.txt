This solution code contains:
	03_a-SVM: Support Vector Machine approach, 
		  this script trains the classfifier and select the best classifier. 
		  
		ex5.py
		output: The best classfifier is saved as: svm_trainned_kernel
			The predicted labels are saved in: ./resultFiles/minist_test_result.csv
		
		note: You need to provide the training set in the folder ./data.

	03_b-MLP: Multilayer perceptron approach, 
		  this script trains the classfifier and select the best classifier. 
		  
		ex5.py
		output: The best classfifier is saved as: mlp_trainned_kernel
			The predicted labels are saved in: ./resultFiles/minist_test_result.csv
		
		note: You need to provide the training set in the folder ./data.


	Hybrid SVM-MLP: Hybrid approach, 
		  this script takes three different best classifiers and by 
		  simple votation process selects the commom lebel among them. 
		
		ex5-hybrid.py  
		output: 
			The predicted labels are saved in: ./resultFiles/minist_test_result.csv
		
	

