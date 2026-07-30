from bs4 import BeautifulSoup
import pandas as pd
import os
from playwright.sync_api import sync_playwright
from supermarkt.clean import Product, clean_product_dirk
from urllib.parse import urljoin
import random
from tqdm import tqdm

DEBUG = False
base = "https://www.dirk.nl/boodschappen"
category = "vlees-vis"
subcategory = "kip-kalkoen"
NON_FOOD_CATEGORIES = ["/boodschappen/huishoud-huisdieren", "/boodschappen/kind-drogisterij", "/boodschappen/non-food"]

url = os.path.join(base, category, subcategory)
url = "https://www.dirk.nl/boodschappen/vlees-vis/kip-kalkoen/vleeschmeesters%20kipfilet/112179"
url2 = "https://www.dirk.nl/boodschappen/diepvries/ijs/ola-magnum-almond/103754"

class DirkScraper:
    """
    Provides functionality for scraping product information from the Dirk website.
    """
    def __init__(self, url, headless=True, debug=False):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page(
            viewport={"width": 1920, "height": 1080}
        )
        self.food_categories = {}
        self.debug = debug

        self.page.goto(url)


        # Close the welcome banner (Once)
        close = self.page.locator('button[aria-label="sluit melding"]')
        try:
            close.wait_for(state="visible", timeout=3000)
            close.click()
        except:
            pass

    def scrape_categories(self, base_url: str):
        """
        Scrape all product URLs from the website, and store to .csv.
        """
        self.page.goto(base_url)
        html = self.page.content()
        soup = BeautifulSoup(html, "html.parser")

        departments = soup.find_all("a", "department")
        categories = [a["href"] for a in departments]

        food_categories = [cat for cat in categories if cat not in NON_FOOD_CATEGORIES]

        self.food_categories = {cat: [] for cat in food_categories}

        # Sub-categories
        for cat in self.food_categories.keys():
            url = urljoin(base, cat)
            self.page.goto(url)
            html = self.page.content()
            soup = BeautifulSoup(html, "html.parser")

            subcats = soup.select(f'li a[href*="{cat}"]')
            subcat_hrefs = [a["href"] for a in subcats]
            subcat_hrefs_dict = {subcat_href : [] for subcat_href in subcat_hrefs}

            self.food_categories[cat] = subcat_hrefs_dict

            # Individual Product Links
            for subcat in subcat_hrefs:
                url_sub = urljoin(base, subcat)
                self.page.goto(url_sub)
                html = self.page.content()
                soup = BeautifulSoup(html, "html.parser")

                products = soup.select(f'article a.top[href*="{subcat}"]')
                product_hrefs = [a["href"] for a in products]

                self.food_categories[cat][subcat] = product_hrefs
            
                # self.page.wait_for_timeout(random.randint(500, 2500))

        if self.debug:
            print(f"categories: {self.food_categories}")

        rows = []
        for cat, subcats in self.food_categories.items():
            for subcat, products in subcats.items():
                for product in products:
                    rows.append({
                        "cat": cat,
                        "sub_cat": subcat,
                        "product": urljoin(base, product)
                    })

        df = pd.DataFrame(rows)
        df.to_csv("data/raw/dirk_products.csv", index=False)
    
    def get_product_html(self, url: str):
        """Retrieve product HTML after loading dynamic content.

        Opens dropdown sections on the product page to load nutrition
        and ingredient information, then returns the resulting HTML.
        """
        self.page.goto(url)

        has_nutrition, has_ingredients = True, True

        # Open Voedingswaarden
        section = self.page.locator("div.title", has=self.page.locator("h3", has_text="Voedingswaarden"))
        if section.count() == 0: # dropdown not there
            has_nutrition = False
        else:
            section.locator("button").click()
        if self.debug:
            self.page.screenshot(path="debug1.png")

        # Open Ingredients
        section = self.page.locator("div.title", has=self.page.locator("h3", has_text="Ingrediënten"))
        if section.count() == 0: # dropdown not there
            has_ingredients = False
        else:
            section.locator("button").click()
        if self.debug:    
            self.page.screenshot(path="debug2.png")

        if has_nutrition:
            self.page.wait_for_selector("ul.nutrition", state="attached")
        if has_ingredients:
            self.page.wait_for_selector("div.body", state="attached")

        html = self.page.content()

        return html, has_nutrition, has_ingredients

    def _scrape_price(self, soup: BeautifulSoup):
        price_div = soup.find("div", class_="price")
        euros = price_div.find("span", class_="price-large").text.strip()
        cents_tag = price_div.find("span", class_="price-small")

        if cents_tag is not None: # euro.cents
            cents = cents_tag.text.strip()
            price = float(f"{euros}.{cents}") # TODO: if there's only cents
        else: # cents only
            cents = euros
            price = float(f"{0}.{cents}")

        title_div = soup.find("div", class_="title")
        title = title_div.find("h1").text.strip()

        portion = soup.find("p", class_="subtitle").text.strip()

        if self.debug:
            print(f"Scraped price, title, portion: {price}, {title}, {portion}")

        return price, title, portion

    def _scrape_nutrition(self, soup: BeautifulSoup, has_nutrition: bool, has_ingredients: bool):
        nutrition, ingredients = None, None

        # Ingredients scrape
        if has_ingredients:
            article = soup.find("h3", string="Ingrediënten").find_parent("article")
            body = article.find("div", class_="body")
            ingredients_tag = body.find("p")
            if ingredients_tag is not None: # e.g. https://www.dirk.nl/boodschappen/aardappelen-groente-fruit/groente/brandwijk%20groene%20erwten/4543
                ingredients = ingredients_tag.get_text(strip=True)

            if self.debug:
                print("Scraped Ingredients")

        # Nutrition scrape
        if has_nutrition:
            ul = soup.find("ul", class_="nutrition")
            if ul is not None:
                nutrition = {}
                for item in ul.find_all("div", class_="nutrition-item"):
                    spans = item.find_all("span")
                    if len(spans) == 2: # (key, value)
                        nutrition[spans[0].text.strip()] = spans[1].text.strip()

            if self.debug:
                print("Scraped Voedingswaarden")

        return nutrition, ingredients

    def scrape_product(self, url: str, sub_cat=""):
        """
        Scrape product information from a product page.

        Returns:
        A Product object containing the scraped product information.
        """
        html, has_nutrition, has_ingredients = self.get_product_html(url)
        soup = BeautifulSoup(html, "html.parser")

        price, name, portion = self._scrape_price(soup)
        nutrients, ingredients = self._scrape_nutrition(soup, has_nutrition, has_ingredients)

        return Product(
            href=url,
            sub_cat=sub_cat,
            name=name,
            price=price,
            quantity=portion,
            ingredients=ingredients,
            nutrients=nutrients,
        )

    def raw_scrape_to_csv(self, products: list[Product], path="data/raw/dirk.csv", mode="a"):
        df = pd.DataFrame(products)
        df.to_csv(path, mode=mode, index=False)

scraper = DirkScraper(url, headless=True, debug=DEBUG)


def run():
    # Load already processed subcategories
    output_file = "data/raw/dirk.csv"

    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        completed = set(existing["sub_cat"].unique())
    else:
        completed = set()

    df = pd.read_csv("data/raw/dirk_products.csv")
    for subcat, group in df.groupby("sub_cat"):
        if subcat in completed:
            print(f"Skipping {subcat}")
            continue

        print(f"Processing {subcat} ({len(group)} products)")

        products = []

        for href in tqdm(group["product"], desc=subcat, total=len(group)):
            try:
                # scraper.page.wait_for_timeout(random.randint(1000, 2500)) # optional timeout if getting flagged
                product = scraper.scrape_product(url=href, sub_cat=subcat)
                clean_product_dirk(product)
                products.append(product)

            except Exception as e:
                print(f"Failed: {href} - {e}")

        # Save after each subcategory
        if products is not []:
            scraper.raw_scrape_to_csv(products, mode="a")
            print(f"Wrote {subcat} to .csv")
            completed.add(subcat)

if not os.path.exists("data/raw/dirk_products.csv"):
    scraper.scrape_categories(base_url=base)

run()
