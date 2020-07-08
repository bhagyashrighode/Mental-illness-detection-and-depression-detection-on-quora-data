#Quora Web Scrapping
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import csv
from selenium import webdriver

#Csv File to write Scrapped Data
with open('sub10.csv','w') as file:
    file.write("Question")

link1 = input("Enter first link")
#link2 = input("Enter second link")
manylinks = list()
manylinks.append(link1)
#manylinks.append(link2)
for olink in manylinks:
    qlinks = list()    
    browser = webdriver.Chrome(executable_path=r'C:\\Users\\admin\\projects\\firstdjango\\chromedriver.exe')
    browser.get(olink)
    time.sleep(1)
    elem = browser.find_element_by_tag_name("body")


    no_of_pagedowns = 50
    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        no_of_pagedowns-=1
    post_elems =browser.find_elements_by_xpath("//a[@class='question_link']")
    for post in post_elems:
        qlink = post.get_attribute("href")
        print(qlink)
        qlinks.append(qlink)

    for qlink in qlinks:

        append_status=0

        row = list()

        browser.get(qlink)
        time.sleep(1)


        elem = browser.find_element_by_tag_name("body")


        no_of_pagedowns = 1
        while no_of_pagedowns:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
            no_of_pagedowns-=1


        #Question Names
        qname =browser.find_elements_by_xpath("//div[@class='question_text_edit']")
        for q in qname:
            print(q.text)
            row.append(q.text)



        with open('sub10.csv','a') as file:
                writer = csv.writer(file)
                writer.writerow(row)
