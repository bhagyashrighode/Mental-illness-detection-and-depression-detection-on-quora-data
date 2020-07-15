# Mental-illness-detection-and-depression-detection-on-quora-data
Quora Depression and mental illness detection Analysis using Baye's Theorem.

Software Requirement:
	python 3.5
	pycharm/jupiter nootbook/spyder
packages-numpy,pandas,csv,nltk,selenium,webdriver.
	

for this project first important thing is dataset.
For this purpose first we collect the data using scraping quora using file linkscrape.py

To run file linkscrape.py
Install Selenium,webdriver pakage in your python

Output of linscrape file-

Enter first link>? Enter the link of your quora question here

Using this we collected the data file submission,submission1,submission2,submission3,submission4,submission5,submission6,submission7,submission8,submission9,submission10
question related to depression and mental illness topic.
As well as collocted normal nondepressed data as sub1,sub2,sub3,sub4,sub5,sub6,sub7,sub8,sub9,sub10.

For this project we need depressed as well as non-depressed data so combined all data in new file Book11 question,label-1 depressed,label-0 non depressed.


Now devloping model useing bayes theroem which detect the depressed and non depressed question.

To run classifier_native_bayes.py

Install nltk,pandas,numpy,csv packages to run this file

After that output-
Enter Question-Enter your question here
Sentiment-Depressed/Non-Depressed.

In this way we can find depressed and non depressed sentiment.






