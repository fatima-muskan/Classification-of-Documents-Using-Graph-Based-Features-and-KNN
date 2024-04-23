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

    urls = ['https://www.daraz.pk/catalog/?spm=a2a0e.searchlist.pagination.1.33264161OuLtGW&_keyori=ss&from=input&q=disease',
            'https://www.daraz.pk/catalog/?spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho&_keyori=ss&from=input&q=disease&page=2',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=4&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=3&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=5&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=6&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=7&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=8&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=9&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=10&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=11&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=12&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=13&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=14&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho',
            'https://www.daraz.pk/catalog/?_keyori=ss&from=input&page=15&q=disease&spm=a2a0e.searchlist.pagination.2.6aff4161B8Fbho']
    all_titles = []

    for idx, url in enumerate(urls):
        driver.get(url)
        AutoScroll(driver)

        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        titles = soup.find_all('div', class_='title-wrapper--IaQ0m')

        titles_text = [title.text.strip() for title in titles]
        all_titles.append(titles_text)

        # Writing titles to a separate text file
        with open(f"disease{idx+1}.txt", "w", encoding="utf-8") as textfile:
            for title in titles_text:
                textfile.write(title + "\n")

        print("Titles scraped from", url, ":", titles_text)

    driver.quit()

    with open("disease_data.csv", "w", newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        for titles in all_titles:
            csvwriter.writerow(titles)

    print("Total titles scraped:", sum(len(titles) for titles in all_titles))
