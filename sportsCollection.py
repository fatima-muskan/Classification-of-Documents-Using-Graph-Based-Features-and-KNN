import csv
from bs4 import BeautifulSoup
from selenium import webdriver
import math
import time

def AutoScroll(driver):
    z = 1000
    previous_height = -math.inf
    while True:
        z += 1000
        current_height = driver.execute_script("return document.documentElement.scrollHeight")
        if current_height == previous_height:
            break
        previous_height = current_height
        scroll = "window.scrollTo(0," + str(z) + ")"
        driver.execute_script(scroll)
        time.sleep(5)
        z += 1000


if __name__ == "__main__":
    driver = webdriver.Chrome()

    urls = ['https://sports.yahoo.com/soccer/',
            'https://sports.yahoo.com/nhl/',
            'https://sports.yahoo.com/boxing/',
            'https://sports.yahoo.com/cycling/',
            'https://sports.yahoo.com/horse-racing/',
            'https://sports.yahoo.com/golf/',
            'https://sports.yahoo.com/nfl/',
            'https://sports.yahoo.com/motorsports/',
            'https://sports.yahoo.com/ufl/',
            'https://sports.yahoo.com/mlb/',
            'https://sports.yahoo.com/nba/',
            'https://sports.yahoo.com/mma/',
            'https://sports.yahoo.com/college-football/',
            'https://sports.yahoo.com/college-basketball/',
            'https://sports.yahoo.com/college-womens-basketball/']
    all_titles = []

    for idx, url in enumerate(urls):
        driver.get(url)
        AutoScroll(driver)

        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        titles = soup.find_all('p', class_='Fz(14px) Lh(19px) Fz(13px)--sm1024 Lh(17px)--sm1024 LineClamp(2,38px) LineClamp(2,34px)--sm1024 M(0)')

        titles_text = [title.text.strip() for title in titles]
        all_titles.append(titles_text)

        print("Titles scraped from", url, ":", titles_text)
         # Writing titles to a separate text file
        with open(f"sports{idx+1}.txt", "w", encoding="utf-8") as textfile:
            for title in titles_text:
                textfile.write(title + "\n")

        print("Titles scraped from", url, ":", titles_text)

    driver.quit()

    with open("sports.csv", "w", newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        for titles in all_titles:
            csvwriter.writerow(titles)

    print("Total titles scraped:", sum(len(titles) for titles in all_titles))
