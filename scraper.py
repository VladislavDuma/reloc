import asyncio
import json
import random
import re
import aiohttp
import aiofiles
import aiofiles.os
import yaml
import os

from bs4 import BeautifulSoup
from datetime import datetime


# -- Scraper class -----------------------------------------------------------------------------------------------------
class Scraper:
    def __init__(self, config='./configs/scraper_config_base.yaml'):
        """
        Инициализация класса Scraper. При инициализации необходимо указать конфигурацию для запуска.

        :param str config: Путь к конфигурационному файлу для скрапера.
        """
        scraper_config = None
        with open(config, 'r') as stream:
            try:
                scraper_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.scraper_config = scraper_config
        self.base_path = f'./data/{scraper_config["config_folder"]}'
        self.path_to_site_configs = f'./{scraper_config["root_folder"]}/{scraper_config["config_folder"]}'
        self.site_list = scraper_config['site_list']

    # session = aiohttp.ClientSession()
    # session = aiohttp.ClientSession(trust_env=True)
    # session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))

    # async with current_session.get(url=url, headers=headers) as response:

    async def download(self, url=None, site_config=None, main_page=False):
        """
        Скачивает HTML-странницу переданной в *site_config*.

        :param str url: Ссылка которую необходимо скачать и сохранить
        :param dict site_config: Конфигурация сайта
        :param bool main_page: Флаг, который принимает значение True или False, указывая,
            является ли ссылка главной страницей сайта или нет
        :return: В случае корректно скачанной главной страницы сайта, возвращает папку
            с названием сайта и путь к скачанному файлу.
        """
        if url is None:
            url = site_config['url']
        base_folder = self.scraper_config['config_folder']
        headers = {'User-Agent': self.scraper_config['user-agent']}

        session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        async with session as current_session:
            async with current_session.get(url=url, headers=headers, ssl=False) as response:
                html_response = await response.text()

        # Фильтруем данные от http/https и скобок в названии для сохранения в файловой системе
        http_filter_list = ['https://', 'http://', '/', ' ']
        for hfl in http_filter_list:
            url = url.replace(hfl, '')

        if response.status == 200:
            # Указываем путь и название файла, где будет хранится скачанная HTML страница
            update_time = datetime.now().strftime('%H%M_%d%m%Y')
            current_date = datetime.now().strftime('%d%m%Y')
            if main_page:
                path = f'./data/{base_folder}/{site_config["folder"]}/main_page/{url}_{update_time}.html'
            else:
                path = f'./data/{base_folder}/{site_config["folder"]}/{current_date}/{url}.html'

            # Создаём папки для скачиваемых файлов и убеждаемся в их корректности
            parent = os.path.dirname(path)
            if not await aiofiles.os.path.exists(parent):
                await aiofiles.os.makedirs(parent)
            if not await aiofiles.os.path.isdir(parent):
                raise ValueError('Пожалуйста, убедитесь что указан каталог, а не файл')

            # Сохраняем полученный результат
            async with aiofiles.open(path, 'w', encoding='utf-8') as output_file:
                await output_file.write(html_response)
            return path
        else:
            print(response.status)
        # except (
        #         ConnectionResetError,
        #         aiohttp.ClientOSError,
        #         aiohttp.ServerDisconnectedError,
        #         aiohttp.ClientConnectorError
        # ):
        #     print(f'Try to reconnect for URL: {url} - after {300} seconds')
        #     return None

    async def start_page_refresh(self, end_time):
        """
        Запускает цикличное обновление главных страниц сайтов, указанных в ./configs/scraper_config_*.yaml

        :param datetime end_time: Время окончания обновлений после запуска скрапера.
        """
        if self.site_list is not None:
            tasks = []
            for site in self.site_list:
                # Чтение конфигурации скрапера
                site_config = None
                with open(f'./configs/{self.path_to_site_configs}/{site}.yaml') as stream:
                    try:
                        site_config = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                site_config = site_config.get('site_config')

                tasks.append(asyncio.create_task(
                    self.cycle_page_update(site_config=site_config, end_time_for_updates=end_time)))

            # Запуск цикличных обновлений
            await asyncio.gather(*tasks)
        print(end_time)

    async def cycle_page_update(self, site_config, end_time_for_updates):
        """
        Асинхронная функция для цикличного обновления отдельно взятого сайта.

        :param dict site_config: Конфигурация сайта (папка для сохранения файла, теги, интервал между обновлениями)
        :param datetime end_time_for_updates: Время окончания обновлений
        """
        # Запуск цикличных обновлений
        update_again = True
        while update_again:
            update_again = await self.page_update(site_config=site_config,
                                                  end_time_for_updates=end_time_for_updates)
            # Время обновления главной страницы
            time_to_update = site_config.get('time_update', 600)
            refresh_time = int(time_to_update['hours']) * 3600 + \
                           int(time_to_update['minutes']) * 60 + \
                           int(time_to_update['seconds'])

            if update_again:
                await asyncio.sleep(refresh_time)
        print(f'Page: {site_config["url"]} | Time: {datetime.now()} | {update_again}')

    async def page_update(self, site_config, end_time_for_updates):
        # Загрузка/обновление главной страницы сайта
        main_page = await self.download(site_config=site_config,
                                        main_page=True)

        extracted_links = await self.extract_list_of_pages(main_page=main_page,
                                                           site_config=site_config)
        # Скачивание новых страниц
        await asyncio.sleep(1)
        for i, link in enumerate(extracted_links):
            time_to_sleep = random.randint(300, 800) / 100
            print(f'Page number: {i + 1}\nTime to sleep: {time_to_sleep} \nPage: {link}')
            await asyncio.sleep(time_to_sleep)
            await self.download(url=link,
                                site_config=site_config,
                                main_page=False)
        print(f'Main page site: {main_page}\nCurrent time: {datetime.now()}')
        current_time = datetime.now()
        if current_time < end_time_for_updates:
            return True
        else:
            return False

    async def extract_list_of_pages(self, main_page, site_config):
        """
        Извлекает список HTML-страниц, которые необходимо скачать. Ранее загруженные
        страницы фильтруются и не загружаются повторно.

        :param str main_page: Путь к главной странице сайта, сохранённая локально.
        :param dict site_config: Конфигурация сайта (папка для сохранения файла, теги, интервал между обновлениями).
        """
        # Читаем полученный результат
        # if main_page:
        async with aiofiles.open(main_page, 'r', encoding='utf8') as output_file:
            page = await output_file.read()
        soup = BeautifulSoup(page, 'lxml')

        # Основные параметры сайта для извлечения ссылок
        url = site_config.get('url', None)
        tag = site_config.get('tag', 'a')
        regular = site_config.get('regular', None)
        tag_class = site_config.get('tag_class', None)
        add_page_name = site_config.get('add_page_name', False)
        https_line = site_config.get('https_line', True)
        delete_part_of_url = site_config.get('delete_part_of_url', None)

        # Извлечение необходимых ссылок
        filtered_href_list = []
        if regular is not None and tag_class is not None:
            # С использованием регулярных выражений и тега класса в HTML странице
            soup_result = soup.find_all(tag, tag_class)
            href_list = [r.get('href') for r in soup_result if r.get('href') is not None]
            for hl in href_list:
                temp = re.findall(regular, hl)
                if temp:
                    filtered_href_list.append(temp[0])
        elif regular is not None:
            # С использованием регулярных выражений
            soup_result = soup.find_all(tag)
            href_list = [r.get('href') for r in soup_result if r.get('href') is not None]
            for hl in href_list:
                temp = re.findall(regular, hl)
                if temp:
                    filtered_href_list.append(temp[0])
        elif tag_class is not None:
            # С использованием тега класса в HTML странице
            soup_result = soup.find_all(tag, tag_class)
            href_list = [r.get('href') for r in soup_result if r.get('href') is not None]
            filtered_href_list = href_list.copy()

        # Удаление "лишней" части ссылки (случай с [lenta.ru: /parts/news/])
        if delete_part_of_url:
            url = url.replace(delete_part_of_url, '')

        # Добавление исходной ссылки к отфильтрованным, при необходимости.
        # Например, https://lenta.ru/parts/news и /2020/10/03/something.
        # Необходимо для [lenta.ru/parts/news]
        if add_page_name:
            if https_line:
                filtered_href_list = [f'{url}{fhl}' for fhl in filtered_href_list]
            else:
                filtered_href_list = [f'{fhl}' for fhl in filtered_href_list]

        # Преобразуем к множеству, для удаления дубликатов
        filtered_href_set = set(filtered_href_list)

        # Исключаем загруженные раннее ссылки, список которых хранится в файле site_data.json
        # для каждого отдельно взятого сайта
        site_name = site_config.get('folder', None)
        path_to_json = f'{self.base_path}/{site_name}/main_page/site_data.json'
        if await aiofiles.os.path.exists(path_to_json) is True:
            async with aiofiles.open(path_to_json, 'r', encoding='utf-8') as f:
                site_data = await f.read()
            site_data = json.loads(site_data)

            temp_list = site_data['new_page_list'] + site_data['page_list']
            page_list = set(temp_list)
            page_to_download = list(filtered_href_set.difference(page_list))
            page_list = list(page_list)
        else:
            page_list = []
            page_to_download = list(filtered_href_set)

        await self.update_page_list(path_to_json=path_to_json,
                                    main_page_path=main_page,
                                    page_list=page_list,
                                    page_to_download=page_to_download)
        return page_to_download
        # else:
        #     return None

    async def update_page_list(self, path_to_json, main_page_path, page_list, page_to_download):
        """
        Обновление списка ссылок ранее скачанных и ссылок, которые необходимо скачать.
        Информация о скачанных HTML-страницах сохраняется в файле site_data.json,
        рядом со скачанными главными страницами.

        :param path_to_json: Путь к файлу с данными сайта по ссылкам (загружены/не загружены).
        :param main_page_path: Путь, по которому хранится главная страница сайта со ссылками.
        :param page_list: Список ранее загруженных ссылок.
        :param page_to_download: Список ссылок, которые необходимо скачать
        """
        json_data = {
            "updated_main_page": main_page_path,
            "new_page_list": page_to_download,
            "page_list": page_list
        }
        if await aiofiles.os.path.exists(path_to_json) is True:
            async with aiofiles.open(path_to_json) as f:
                content = await f.read()
            template = json.loads(content)
            os.remove(template['updated_main_page'])

        # Сохранение в JSON
        json_data = json.dumps(json_data)
        async with aiofiles.open(path_to_json, mode='w', encoding='utf-8') as output_file:
            await output_file.write(json_data)


if __name__ == "__main__":
    scrap = Scraper('./configs/scraper_config_base.yaml')
    # Задаём время окончания обновлений
    now = datetime.now()
    year = int(now.strftime('%Y'))
    month = int(now.strftime('%m'))
    day = int(now.strftime('%d'))
    hour = 17
    minute = 20
    end_time = datetime(year, month, day, hour, minute)

    # Запуск асинхронных функций
    asyncio.run(scrap.start_page_refresh(end_time=end_time))
