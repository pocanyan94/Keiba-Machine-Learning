from datetime import datetime
import re
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import tweepy

from modules.constants._local_paths import LocalPaths
from modules.constants._master import Master


class PostNoteX:

    @staticmethod
    def content_text(yosou: list, kaime: list, kaikata: str) -> str:
        """
        コンテンツテキストを生成する。

        Args:
            yosou (list): 予想のリスト。
            kaime (list): 買い目のリスト。
            kaikata (str): 推奨内容。

        Returns:
            str: フォーマットされたコンテンツテキスト。
        """
        content_template = """
        ### 馬印
        {yosou_list}

        ### 買い目
        {kaime_list}

        ### 推奨
        {kaikata}
        """

        # 馬印と買い目のリストを文字列に変換
        yosou_list = "\n".join([f"- {y}" for y in yosou])  # 予想リスト
        kaime_list = "\n".join([f"- {k}" for k in kaime])  # 買い目リスト

        # コンテンツを生成
        content = content_template.format(
            yosou_list=yosou_list, kaime_list=kaime_list, kaikata=kaikata
        )

        return content

    @staticmethod
    def post_to_note(
        email: str,
        password: str,
        title: str,
        markdown_content: str,
        image_path: str = None,
    ) -> str:
        """
        noteに記事を投稿する。

        Args:
            : noteのメールアドレス。
            : noteのパスワード。
            : 記事のタイトル。
            markdown_content: 記事のMarkdownコンテンツ。
            image_path: アップロードする画像のパス。初期値None

        Returns:
            str: 記事のURL。
        """
        print("1. noteにログイン中...")
        session = PostNoteX.get_note_session(email, password)

        print("2. 記事を作成中...")
        # MarkdownをHTMLに変換（簡易版）
        html_content = PostNoteX.markdown_to_html(markdown_content)
        article_id, article_key = PostNoteX.create_article(session, title, html_content)

        if not article_id:
            return False

        if image_path:
            print("3. 画像をアップロード中...")
            image_key, image_url = PostNoteX.upload_image(session, image_path)

        print("4. 記事を下書き保存中...")
        success = PostNoteX.update_article_draft(
            session, article_id, title, html_content, image_key
        )

        if success:
            print("\n✅ 投稿完了！")
            print(f"記事URL: https://note.com/keiba_ai_poca/n/{article_key}")
        else:
            print("\n✅ 投稿失敗...")

        return f"https://note.com/keiba_ai_poca/n/{article_key}"

    @staticmethod
    def create_article(
        session: requests.Session, title: str, html_content: str
    ) -> tuple:
        """
        新しい記事を作成する。

        Args:
            session : noteへのログインセッション。
            title: 記事のタイトル。
            html_content: 記事のHTMLコンテンツ。

        Returns:
            tuple: 記事のIDとキー。
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/139.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }

        data = {
            "body": html_content,
            "name": title,
            "template_key": None,
        }

        response = session.post(
            "https://note.com/api/v1/text_notes", headers=headers, json=data
        )

        if response.status_code == 201:
            result = response.json()
            article_id = result["data"]["id"]
            article_key = result["data"]["key"]
            print(f"記事作成成功！ID: {article_id}")
            return article_id, article_key
        else:
            print(f"記事作成失敗: {response.status_code}, レスポンス: {response.text}")
            return None, None

    @staticmethod
    def upload_image(session: requests.Session, image_path: str) -> tuple:
        """
        画像をアップロードする。

        Args:
            session: noteへのログインセッション。
            image_path: アップロードする画像のパス。

        Returns:
            tuple: 画像のキーとURL。
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/139.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }

        with open(image_path, "rb") as f:
            files = {"file": f}

            response = session.post(
                "https://note.com/api/v1/upload_image", headers=headers, files=files
            )

        if response.status_code == 201:
            result = response.json()
            image_key = result["data"]["key"]
            image_url = result["data"]["url"]
            print(f"画像アップロード成功！KEY: {image_key}")
            return image_key, image_url
        else:
            print(
                f"画像アップロード失敗: {response.status_code}, レスポンス: {response.text}"
            )
            return None, None

    @staticmethod
    def update_article_draft(
        session: requests.Session,
        article_id: str,
        title: str,
        html_content: str,
        image_key: str = None,
    ) -> bool:
        """
        記事を更新して下書きとして保存する。

        Args:
            session: noteへのログインセッション。
            article_id : 更新する記事のID。
            title: 記事のタイトル。
            html_content: 記事のHTMLコンテンツ。
            image_key: アイキャッチ画像のキー。

        Returns:
            bool: 更新が成功したかどうか。
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/139.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }

        data = {
            "body": html_content,
            "name": title,
        }

        # アイキャッチ画像がある場合は追加
        if image_key:
            data["eyecatch_image_key"] = image_key

        response = session.post(
            f"https://note.com/api/v1/text_notes/draft_save?id={article_id}&is_temp_saved=true",
            headers=headers,
            json=data,
        )

        if response.status_code == 201:
            print("記事の下書き保存成功！")
            return True
        else:
            print(
                f"記事の更新失敗: {response.status_code}, レスポンス: {response.text}"
            )
            return False

    @staticmethod
    def get_note_session(email: str, password: str) -> requests.Session:
        """
        noteにログインしてCookieを取得しSessionを返却する。

        Args:
            email: noteのメールアドレス。
            password: noteのパスワード。

        Returns:
            requests.Session: ログインセッション。
        """
        driver = webdriver.Chrome()

        try:
            # ログインページにアクセス
            driver.get("https://note.com/login")
            time.sleep(1)

            # メールアドレスとパスワードを入力
            email_input = driver.find_element(By.ID, "email")  # IDを使用
            email_input.send_keys(email)
            time.sleep(1)

            password_input = driver.find_element(By.ID, "password")  # IDを使用
            password_input.send_keys(password)

            # ログインボタンを待ち、スクロールしてからクリック
            login_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//button[@type='button' and normalize-space(.)='ログイン']",
                    )
                )
            )

            # スクロールしてボタンを画面内に表示
            driver.execute_script("arguments[0].scrollIntoView();", login_button)

            # ログインボタンをクリック
            login_button.click()

            # ログイン完了を待つ
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "o-login__form"))
            )
            time.sleep(5)

            # Cookieを取得
            cookies = driver.get_cookies()

            session = requests.Session()
            # クロームドライバーから取得したクッキーをrequestsに設定
            for cookie in cookies:
                session.cookies.set(cookie["name"], cookie["value"])

            return session

        finally:
            driver.quit()

    @staticmethod
    def markdown_to_html(markdown_text: str) -> str:
        """
        簡易的なMarkdownをHTMLに変換する。

        Args:
            markdown_text: Markdown形式のテキスト。

        Returns:
            str: HTML形式のテキスト。
        """
        # テキストの整形
        html = markdown_text.strip()

        # 見出しの変換
        html = re.sub(r"(?m)^\s*### (.+)$", r"<h3>\1</h3>", html)
        html = re.sub(r"(?m)^\s*## (.+)$", r"<h2>\1</h2>", html)
        html = re.sub(r"(?m)^\s*# (.+)$", r"<h1>\1</h1>", html)

        # リスト項目を<li><p>に変換（<li>自体を<p>で囲まない）
        html = re.sub(r"(?m)^\s*\- (.+)$", r"<li><p>\1</p></li>", html)

        # 強調
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

        # コードブロック
        html = re.sub(
            r"```(.*?)```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL
        )
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)

        # <h2>の後に続く<li>を<ul>で囲む
        html = re.sub(
            r"(<h3>.*?</h3>)\s*((?:<li>.*?</li>\s*)+)",
            lambda m: f"""{m.group(1)}<ul>
                {"".join(m.group(2).strip().splitlines())}
                </ul>""",
            html,
            flags=re.DOTALL,
        )

        # 段落を生成する際、見出しとリストが正しく処理されるように
        paragraphs = html.split("\n")
        html = "\n".join(
            [
                (
                    f"<p>{p.strip()}</p>"
                    if not (
                        p.startswith("<h") or p.startswith("<ul") or p.startswith("<l")
                    )
                    and p.strip()
                    else p
                )
                for p in paragraphs
            ]
        )

        return html.strip()

    @staticmethod
    def login_x(
        X_API_KEY: str,
        X_API_SECRET_KEY: str,
        X_ACCESS_TOKEN: str,
        X_ACCESS_TOKEN_SECRET: str,
        X_BEARER_TOKEN: str,
    ) -> tweepy.Client:
        """
        Twitter APIにログインし、認証されたクライアントを返す。

        Args:
            X_API_KEY (str): Twitter APIのコンシューマーキー。
            X_API_SECRET_KEY (str): Twitter APIのコンシューマーシークレットキー。
            X_ACCESS_TOKEN (str): Twitter APIのアクセストークン。
            X_ACCESS_TOKEN_SECRET (str): Twitter APIのアクセストークンシークレット。
            X_BEARER_TOKEN (str): Twitter APIのベアラートークン。

        Returns:
            tweepy.Client: 認証されたTwitter APIクライアント。

        Raises:
            Exception: 認証に失敗した場合に例外を投げる。
        """
        # APIキーとトークンを設定

        # 認証を行う
        auth = tweepy.OAuthHandler(X_API_KEY, X_API_SECRET_KEY)
        auth.set_access_token(X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET)

        # APIオブジェクトを作成
        api = tweepy.API(auth)

        # 認証が成功したか確認
        try:
            api.verify_credentials()
            print("認証に成功しました。")
        except Exception as e:
            print("認証に失敗しました。", e)

        client = tweepy.Client(
            bearer_token=X_BEARER_TOKEN,
            consumer_key=X_API_KEY,
            consumer_secret=X_API_SECRET_KEY,
            access_token=X_ACCESS_TOKEN,
            access_token_secret=X_ACCESS_TOKEN_SECRET,
        )

        return client

    @staticmethod
    def race_result_processing(today: str, shutuba_data_merger) -> tuple:
        """
        出馬情報を処理し、必要な情報を取得する。

        Args:
            today (str): 日付（YYYYMMDD形式）。
            shutuba_data_merger: 出馬データを持つマージオブジェクト。

        Returns:
            tuple: 開催地、レース番号、レース名、レースタイプ、コース長。
        """

        def fullwidth_to_halfwidth(text: str) -> str:
            """全角数字を半角数字に変換するヘルパー関数。"""
            fullwidth_digits = str.maketrans(
                {
                    chr(i): chr(i - 0xFEE0) for i in range(0xFF10, 0xFF20)
                }  # 全角0-9を対応する半角に変換
            )
            return text.translate(fullwidth_digits)

        date = datetime.strptime(today, "%Y%m%d")
        japanese_weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        weekday_index = date.weekday()  # 月曜日が0、日曜日が6
        short_weekday = japanese_weekdays[weekday_index]
        date = date.strftime("%Y/%m/%d") + f"({short_weekday})"

        reverse_place_dict = {v: k for k, v in Master.PLACE_DICT.items()}
        kaisai = (
            shutuba_data_merger._population_df["kaisai"].map(reverse_place_dict).iloc[0]
        )
        race_num = (
            shutuba_data_merger._population_df["race_id_new"].iloc[0][-2:].lstrip("0")
        )

        shutsuba = pd.read_csv(
            LocalPaths.TARGET_SHUTHUBA_PATH,
            encoding="shift_jis",
            low_memory=False,
            dtype={"レースID(新)": "object"},
        )
        race_name = fullwidth_to_halfwidth(shutsuba["略レース名"].iloc[0])
        race_type = shutsuba["芝・ダート"].iloc[0]
        course_len = shutsuba["距離"].iloc[0]

        return kaisai, race_num, race_name, race_type, course_len
