# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:23:45 2023

@author: mouts
"""

import ScraperFC as sfc
import traceback

for year in range(2024,2025):
    scraper = sfc.Capology()
    try:
        salaries_per_year = scraper.scrape_payrolls(year=year, league="EPL", currency='eur')
    except:
        traceback.print_exc()
    finally:
        scraper.close()
    
    salaries_per_year.to_csv(f'Salaries\\epl_salaries_{year}.csv', index=False)