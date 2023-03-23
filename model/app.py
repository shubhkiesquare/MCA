from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import warnings
from requests import session
import datetime

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from requests import Session
from tempfile import mkdtemp
import psycopg2
import numpy as np
import psycopg2.extras as extras
import pandas as pd
import psycopg2 
s = Session()


gng= "https://www.amazon.in/gp/bestsellers/grocery/ref=zg_bs_unv_grocery_1_4861584031_3"

ctnb = "https://www.amazon.in/gp/bestsellers/grocery/4859478031/ref=zg_bs_nav_grocery_1"

cff = "https://www.amazon.in/gp/bestsellers/grocery/4860057031/ref=zg_bs_nav_grocery_2_4859478031"

wcb = "https://www.amazon.in/gp/bestsellers/grocery/27345411031/ref=zg_bs_nav_grocery_3_4860057031"

rst = "https://www.amazon.in/gp/bestsellers/grocery/27345415031/ref=zg_bs_nav_grocery_4_27345411031"

unrst = "https://www.amazon.in/gp/bestsellers/grocery/27345414031/ref=zg_bs_nav_grocery_4_27345415031"

hnpc = "https://www.amazon.in/gp/bestsellers/hpc/ref=zg_bs_nav_0"

bathnsh = "https://www.amazon.in/gp/bestsellers/hpc/1374276031/ref=zg_bs_nav_hpc_1"

healthc = "https://www.amazon.in/gp/bestsellers/hpc/1374494031/ref=zg_bs_nav_hpc_1"

oralc = "https://www.amazon.in/gp/bestsellers/hpc/1374620031/ref=zg_bs_nav_hpc_1"

skinc = "https://www.amazon.in/gp/bestsellers/hpc/1374407031/ref=zg_bs_nav_hpc_1"

productURL="https://www.amazon.in/Asitis-Nutrition-Natural-Unprocessed-Unroasted/dp/B089Q51FV2/ref=zg_bs_27345414031_1/262-2161519-2520850?pd_rd_i=B089Q51FV2&psc=1"







def get_add_info_soup(URL): 
    
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-gpu")
   
    driver = webdriver.Chrome(executable_path='/home/ubuntu/Notebooks/Web_Automations/chromedriver',
                              options=options)
    driver.get(URL)
    
    
    # this is just to ensure that the page is loaded
    time.sleep(3) 

    html = driver.page_source

    # this renders the JS code and stores all of the information in static HTML code.  
    # Now, we could simply apply bs4 to html variable
    #     soup = BeautifulSoup(html, "html.parser")
    soup = BeautifulSoup(html, "html.parser")
    # soup1 = BeautifulSoup(html, "html.parser")
    driver.quit()
    
    return(soup)
    
def get_prod_info_soup(url):
    r = s.get(url, timeout=5)
    soup = BeautifulSoup(r.text, 'html.parser')
    return(soup)
    
    
def extract_product_info(url, df):
    import datetime
    # asd = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [],
    #                    'brand' : [], 'net_quantity' : [], 'asin' : [], 'Weight' : [], 'manufacturer': []})
    
    # r = s.get(url, timeout=5)
    # soup = BeautifulSoup(r.text, 'html.parser')
    soup = get_prod_info_soup(url)
    
    if soup:
        items = soup.find_all("div", class_="a-cardui _cDEzb_grid-cell_1uMOS expandableGrid p13n-grid-content")

    #     cat = row.Topics.replace(" ", "")

        for item in items:

            ######## getting todays date ####################
            date = datetime.date.today()

            if len(item.find('span', class_='zg-bdg-text').text) is not 0:
                
                # print(list(item.find('span', class_='zg-bdg-text')))
                rank = list(item.find('span', class_='zg-bdg-text'))

                if (item.a) is not None:
                    # print(list(item.find('span', class_='a-size-small')))
                    name = ''.join(item.a.img['alt'])
                else:
                    name = 'NULL'
                    
                # print(str(item.a.img['alt']))
                name = ''.join(item.a.img['alt'])

                if item.find('span', class_='a-size-small') is not None:
                    # print(list(item.find('span', class_='a-size-small')))
                    review = list(item.find('span', class_='a-size-small'))
                else:
                    review = 'NULL'

                if item.find('span', class_="a-icon-alt") is not None:
                    # print(list(item.find('span', class_="a-icon-alt")))
                    rating = list(item.find('span', class_="a-icon-alt"))
                    # print('\n', end='')
                else:
                    rating = 'NULL'
                    # print('\n', end='')

                ######## getting product price ####################

                if item.find('span', class_='p13n-sc-price') is None:
                    # print(list(item.find('span', class_="a-icon-alt")))
                    price = 'NULL'
                    # print('\n', end='')
                else:
                    price = item.find('span', class_="p13n-sc-price").text
                    # print('\n', end='')
            

                ### for additional info 
                prod_url = "https://www.amazon.in/" + item.find('a', class_="a-link-normal", href= True)['href']
                brand, net_quant, Weight, asin, manufacturer = add_product_info(prod_url)
                print(rank, brand, net_quant, Weight, asin, manufacturer)
                
            else:
                print("SKU Skipped", )
                rank = np.NAN
                name = np.NAN
                review = np.NAN
                rating = np.NAN
                price = np.NAN
                brand = np.NAN
                net_quant = np.NAN
                asin = np.NAN
                Weight = np.NAN
                manufacturer = np.NAN



            df = df.append({'date' : date, 'product_rank' : rank,'product_name' : name,'product_total_review' : review, \
                                                 'product_total_rating': rating, 'product_Price': price,
                           'brand' : brand, 'net_quantity' : net_quant, 'asin' : asin, 'Weight' : Weight,
                           'manufacturer' : manufacturer}, ignore_index = True)

        df['product_total_review'] = df['product_total_review'].apply(lambda x:str(x).strip("'[]'"))
        df.loc[df['product_total_review'] == 'NULL', 'product_total_review'] = np.NAN
        # df['product_total_review'] = df['product_total_review'].astype(int)
        df['product_rank'] = df['product_rank'].apply(lambda x:str(x).strip("'[#]'"))
        df['product_total_rating'] = df['product_total_rating'].apply(lambda x:str(x).strip("'[ out of 5 start]'"))
        df.loc[df['product_total_rating'] == 'NULL', 'product_total_rating'] = np.NAN
        df['product_total_rating'] = df['product_total_rating'].astype(float)
        df['product_Price'] = df['product_Price'].apply(lambda x:str(x).strip("'₹'"))
        df.loc[df['product_Price'] == 'NULL', 'product_Price'] = np.NAN
        df['product_availibity'] = 1
        df.loc[df['product_Price'].isna(), 'product_availibity'] = 0
    else:
        extract_product_info(url, df)
    return(df)


def get_add_info_1(soup):
    
    info_object = soup.find_all('th', class_="a-color-secondary a-size-base prodDetSectionEntry")
    info_value = soup.find_all('td', class_ = "a-size-base prodDetAttrValue")

    df_add_info = pd.DataFrame({'object' : [],  'value' : []})
    # print(df_add_info)
    for (obj, value) in zip(info_object, info_value):
        object_ = obj.text.replace(" ","")

        value_ = value.text
        value_ = value.text.replace("\n","")


        df_add_info = df_add_info.append({'object' : object_,  'value' : value_}, ignore_index = True)


    ### Brand #############
    if "Brand" in df_add_info['object'].values:
        Brand = df_add_info[df_add_info['object'] == 'Brand']['value'].item().strip()
        Brand = Brand.replace("\u200e", "")
    else:
        Brand = 'Not_Available'    


    ### NetQuantity ########
    if "NetQuantity" in df_add_info['object'].values:
        NetQuantity     = df_add_info[df_add_info['object'] == 'NetQuantity']['value'].item().strip()

    else:
        NetQuantity = 'Not_Available'


    ###### Weight #########
    if "Weight" in df_add_info['object'].values:
        Weight     = df_add_info[df_add_info['object'] == 'Weight']['value'].item().strip()
        Weight = Weight.replace("\u200e", "")
    else:
        Weight = 'Not_Available'


    ### ASIN ############  
    if "ASIN" in df_add_info['object'].values:
        ASIN = df_add_info[df_add_info['object'] == 'ASIN']['value'].item().strip()
        ASIN = ASIN.replace("\u200e", "")

    else:
        ASIN = 'Not_Available'


    ### Manufacturer ########
    if "Manufacturer" in df_add_info['object'].values:
        Manufacturer = df_add_info[df_add_info['object'] == 'Manufacturer'].head(1)['value'].item().strip()
        Manufacturer = Manufacturer.replace("\u200e", "")

    else:
        Manufacturer = 'Not_Available'
        
    return(Brand, NetQuantity, Weight, ASIN, Manufacturer)
    
    
def get_add_info_2(soup):
    
    items = soup.find_all('ul', class_="a-unordered-list a-nostyle a-vertical a-spacing-none detail-bullet-list")

    add_info = []
    for item in items[:1]:
        add_info = (item.text)    

    # add_info = add_info.encode('ascii', 'ignore').decode().replace("\n", "").strip().replace(":","")
    add_info = add_info.replace("\u200e", "").replace("\u200f", "").replace("\n", "").strip().replace(":","")

    df_add_info = pd.DataFrame(add_info.split("  "))
    df_add_info.columns = ['item']

    df_add_info = df_add_info[df_add_info['item']!=""].reset_index(drop = True)
    df_add_info['item'] = df_add_info['item'].str.strip()

    
    ### Brand #############
    if "Brand" in df_add_info['item'].values:
        ind = df_add_info[df_add_info['item'] == 'Brand'].index.tolist()
        Brand = df_add_info.iloc[sum(ind) + 1].item()
    else:
        Brand = 'Not_Available'    


    ### NetQuantity ########
    if "Net Quantity" in df_add_info['item'].values:
        ind = df_add_info[df_add_info['item'] == 'Net Quantity'].index.tolist()
        NetQuantity = df_add_info.iloc[sum(ind) + 1].item()

    else:
        NetQuantity = 'Not_Available'


    ###### Weight #########
    if "Item Weight" in df_add_info['item'].values:
        ind = df_add_info[df_add_info['item'] == 'Item Weight'].index.tolist()
        Weight = df_add_info.iloc[sum(ind) + 1].item()
    else:
        Weight = 'Not_Available'


    ### ASIN ############  
    if "ASIN" in df_add_info['item'].values:
        ind = df_add_info[df_add_info['item'] == 'ASIN'].index.tolist()
        ASIN = df_add_info.iloc[sum(ind) + 1].item()

    else:
        ASIN = 'Not_Available'


    ### Manufacturer ########
    if "Manufacturer" in df_add_info['item'].values:
        ind = df_add_info[df_add_info['item'] == 'Manufacturer'].index.tolist()
        Manufacturer = df_add_info.iloc[ind[0] + 1].item()

    else:
        Manufacturer = 'Not_Available'
    
    return(Brand, NetQuantity, Weight, ASIN, Manufacturer)
    
def get_add_info_2(soup):
    
    items = soup.find_all('ul', class_="a-unordered-list a-nostyle a-vertical a-spacing-none detail-bullet-list")

    if items:
        add_info = []
        for item in items[:1]:
            add_info = (item.text)    

        # add_info = add_info.encode('ascii', 'ignore').decode().replace("\n", "").strip().replace(":","")
        add_info = add_info.replace("\u200e", "").replace("\u200f", "").replace("\n", "").strip().replace(":","")

        df_add_info = pd.DataFrame(add_info.split("  "))
        df_add_info.columns = ['item']

        df_add_info = df_add_info[df_add_info['item']!=""].reset_index(drop = True)
        df_add_info['item'] = df_add_info['item'].str.strip()


        ### Brand #############
        if "Brand" in df_add_info['item'].values:
            ind = df_add_info[df_add_info['item'] == 'Brand'].index.tolist()
            Brand = df_add_info.iloc[sum(ind) + 1].item()
        else:
            Brand = 'Not_Available'    


        ### NetQuantity ########
        if "Net Quantity" in df_add_info['item'].values:
            ind = df_add_info[df_add_info['item'] == 'Net Quantity'].index.tolist()
            NetQuantity = df_add_info.iloc[sum(ind) + 1].item()

        else:
            NetQuantity = 'Not_Available'


        ###### Weight #########
        if "Item Weight" in df_add_info['item'].values:
            ind = df_add_info[df_add_info['item'] == 'Item Weight'].index.tolist()
            Weight = df_add_info.iloc[sum(ind) + 1].item()
        else:
            Weight = 'Not_Available'


        ### ASIN ############  
        if "ASIN" in df_add_info['item'].values:
            ind = df_add_info[df_add_info['item'] == 'ASIN'].index.tolist()
            ASIN = df_add_info.iloc[sum(ind) + 1].item()

        else:
            ASIN = 'Not_Available'


        ### Manufacturer ########
        if "Manufacturer" in df_add_info['item'].values:
            ind = df_add_info[df_add_info['item'] == 'Manufacturer'].index.tolist()
            Manufacturer = df_add_info.iloc[ind[0] + 1].item()

        else:
            Manufacturer = 'Not_Available'
    else:
        Brand = 'Not_Available'
        NetQuantity = 'Not_Available'
        Weight = 'Not_Available'
        ASIN = 'Not_Available'
        Manufacturer = 'Not_Available'
        

    return(Brand, NetQuantity, Weight, ASIN, Manufacturer)
    
def add_product_info(url):
    soup = get_add_info_soup(url)
    
    if soup :
        info_object = soup.find_all('th', class_="a-color-secondary a-size-base prodDetSectionEntry")
        
        if info_object:
            return(get_add_info_1(soup))
        else:
            return(get_add_info_2(soup))
    else :
        add_product_info(url) 
        
def execute_values(conn, df, table):
  
    tuples = [tuple(x) for x in df.to_numpy()]
  
    cols = ','.join(list(df.columns))
    # SQL query to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("the dataframe is inserted")
    cursor.close()
  


 
        
        
        
        
        
        
def handler():
    
  

    df1 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df1 = extract_product_info(gng,df1)
    df1['category'] = 'Grocery & Gourmet Foods'

    df2 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df2 = extract_product_info(ctnb,df2)
    df2['category'] = 'Coffee Tea & Beverages'


    df3 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df3 = extract_product_info(cff,df3)
    df3['category'] = 'Coffee'

    df4 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df4 = extract_product_info(wcb,df4)
    df4['category'] = 'Whole Coffee Beans'


    df5 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df5 = extract_product_info(rst,df5)
    df5['category'] = 'Roasted'



    df6 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df6 = extract_product_info(unrst,df6)
    df6['category'] = 'Unroasted'

    df7 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df7 = extract_product_info(hnpc,df7)
    df7['category'] = 'Health & Personal Care'


    df8 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df8 = extract_product_info(bathnsh,df8)
    df8['category'] = 'Bath & Shower'


    df9 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df9 = extract_product_info(healthc,df9)
    df9['category'] = 'Health Care'


    df10 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df10 = extract_product_info(oralc,df10)
    df10['category'] = 'Oral Care'


    df11 = pd.DataFrame({'date' : [], 'product_rank' : [],'product_name' :[],'product_total_review' :[], 'product_total_rating': [], 'product_Price': [],
                       'brand' : [], 'net_quantity' : [], 'Weight' : [], 'asin' : [], 'manufacturer': []})
    df11 = extract_product_info(skinc,df11)
    df11['category'] = 'skinc Care'



    df_wai = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11], axis = 0)
    
    
    conn = psycopg2.connect(
       database="meta_raw_data", user='scraping_writer', password='sJUZf$TKu3$Z9At3', host='metadb.cluster-cq7xabbes0x8.ap-south-1.rds.amazonaws.com', port= '5432'
    )    
    execute_values(conn,df_wai, 'rawdata.RAW_META_ABSR_DATA')

        
    return("success")


if __name__ == "__main__":
    handler()






















