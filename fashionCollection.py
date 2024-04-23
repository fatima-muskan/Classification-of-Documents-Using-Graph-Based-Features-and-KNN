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

    urls = ['https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-beauty-station/beauty-tools?spm=a2a0e.tm800127096.3456599500.5.248a49200OUhAc&pfilter=1010Badge_206366',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/bags?spm=a2a0e.tm800133185.4591583930.6.7f337b0fCa5Adz',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/winter-wear-western-wear?spm=a2a0e.tm800133185.4591583930.4.7f337b0fTtOvCq&pfilter=1010Badge_202807',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/accessories?spm=a2a0e.tm800133185.4591583930.10.7f337b0f2sarpV',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/hijabs-and-abayas?spm=a2a0e.tm800133185.4591583930.9.7f337b0fTGMD7l&pfilter=1010Badge_202807',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/women-jewellery?spm=a2a0e.tm800133185.4591583930.8.7f337b0fAlNUNh&pfilter=1010Badge_202807',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/jewellery-mens?spm=a2a0e.tm800153859.9605157410.7.4d8f581a9fsMC7&pfilter=1010Badge_202807',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/pret?spm=a2a0e.tm800133185.4591583930.3.7f337b0fULVfwT',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/footwear?spm=a2a0e.tm800133185.4591583930.7.7f337b0fULVfwT',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/formal-wear?spm=a2a0e.tm800153859.9605157410.1.4d8f581apTXxy1&pfilter=1010Badge_202807',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-beauty-station/hair-care?spm=a2a0e.tm800127096.3456599500.7.248a49200OUhAc&pfilter=1010Badge_206366',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/accessories-men?spm=a2a0e.tm800153859.9605157410.10.4d8f581apTXxy1&pfilter=1010Badge_202807',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-beauty-station/makeup?spm=a2a0e.tm800127096.3456599500.3.248a49200OUhAc&pfilter=1010Badge_206366',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/winter-wear?spm=a2a0e.tm800153859.9605157410.6.4d8f581apTXxy1&pfilter=1010Badge_202807',
            'https://pages.daraz.pk/wow/gcp/daraz/channel/pk/-the-fashion-store/footwear-men?spm=a2a0e.tm800153859.9605157410.4.4d8f581apTXxy1&pfilter=1010Badge_202807']
    all_titles = []

    for idx, url in enumerate(urls):
        driver.get(url)
        AutoScroll(driver)

        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        titles = soup.find_all('span', class_='product-item-bottom-title')

        titles_text = [title.text.strip() for title in titles]
        all_titles.append(titles_text)

        # Writing titles to a separate text file
        with open(f"fashion{idx+1}.txt", "w", encoding="utf-8") as textfile:
            for title in titles_text:
                textfile.write(title + "\n")

        print("Titles scraped from", url, ":", titles_text)

    driver.quit()

    with open("fashion_data.csv", "w", newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        for titles in all_titles:
            csvwriter.writerow(titles)

    print("Total titles scraped:", sum(len(titles) for titles in all_titles))
