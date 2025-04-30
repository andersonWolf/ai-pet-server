import anthropic
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import pandas as pd
import re
import tiktoken
from pinecone import Pinecone
from datetime import datetime

load_dotenv()  # 讀取 .env 檔案
# clientGPT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# clientCLAUDE = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
# clientXAI = anthropic.Anthropic(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/",)
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

global_reference_id = 1

class AIChatbot:
    def __init__(self, bot_name, ai_model="gpt-4o"):
        """
        初始化一個 AI 機器人
        :param bot_name: 機器人名稱
        :param ai_model: AI 模型 (預設為 gpt-4o，可選 claude-3-5-sonnet 等)
        :param training_data: 預載的訓練數據
        """
        load_dotenv()
        self.clientGPT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.clientCLAUDE = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.bot_name = bot_name
        self.ai_model = ai_model
        self.messages_history = []  # 獨立的對話歷史
        self.messages_history_ai = []
        self.messages_history_ui = []
        self.message_folder_path = f"./message_record/{self.bot_name}/"
        ensure_directory_exists(self.message_folder_path)
        self.training_message = None
        self.print_prompt = True
    def get_messages_history(self):
        """ 獲取對話紀錄 """
        return self.messages_history
    def read_previous_messages(self, message_file_path=None, messages_history=None):
        messages = []
        if message_file_path:
            if os.path.exists(message_file_path):
                # 讀取 CSV 檔案並轉換成對應的格式
                messages_history_df = pd.read_csv(message_file_path)
                messages = messages_history_df.to_dict(orient='records')
                print(f"讀取之前的歷史訊息：\n{messages}")
            # 檢查最後一筆資料的 role，如果是 'user'，則刪除
            if messages and messages[-1]["role"] == "user":
                messages.pop()
            self.messages_history_ai.extend(messages)
            self.messages_history_ui.extend(messages)
        if messages_history:
            self.messages_history_ai.extend(messages_history)
            self.messages_history_ui.extend(messages_history)
    def read_training_messages(self, message_file_path):
        self.training_message = get_filepath_to_messages(message_file_path)

    def send_message(self, message):
        """
        向 AI 送出訊息並獲取回應
        :param message: 使用者輸入的訊息
        :return: AI 回應
        """
        self.messages_history.append({"role": "user", "content": message})

        # 準備對話內容
        messages = self.messages_history

        # 選擇對應的 AI 模型
        if self.ai_model.startswith("gpt"):
            response = self.clientGPT.chat.completions.create(
                model=self.ai_model,
                messages=messages,
                temperature=0.7
            )
            reply = response.choices[0].message.content

        elif self.ai_model.startswith("claude"):
            response = self.clientCLAUDE.messages.create(
                model=self.ai_model,
                max_tokens=8192,
                temperature=0.7,
                messages=messages
            )
            reply = response.content[0].text

        else:
            reply = "⚠️ 不支援的 AI 模型"

        # 記錄 AI 回應
        self.messages_history.append({"role": "assistant", "content": reply})
        return reply
    def assistant_normal(self, query="", role_context="", task_context="", tone_context="", input_data="", examples="",
                         task_description="", immediate_task="", precognition="", output_formatting="", answer_tag="",
                         return_all_message=False, add_response_to_ui=True):
        ######################################## 提示元素 ########################################
        def prompt_creation():
            ##### 提示元素 1：`user` 角色
            ROLE_CONTEXT = f"""{role_context}"""

            ##### 提示元素 2：任务上下文
            TASK_CONTEXT = f"""{task_context}"""

            ##### 提示元素 3：语气上下文
            TONE_CONTEXT = f"""{tone_context}"""

            ##### 提示元素 4：要处理的输入数据
            INPUT_DATA = f"""<input_data>{input_data}</input_data>"""

            ##### 提示元素 5：示例
            EXAMPLES = f"""{examples}"""

            ##### 提示元素 6：详细的任务描述和规则
            TASK_DESCRIPTION = f"""<query>{query}</query>{task_description}”"""

            ##### 提示元素 7：立即任务描述或请求
            IMMEDIATE_TASK = f"""{immediate_task}"""

            ##### 提示元素 8：预想（逐步思考）
            PRECOGNITION = f"""{precognition}"""

            ##### 提示元素 9：输出格式
            OUTPUT_FORMATTING = f"""
            <{answer_tag}>
            {output_formatting}
            </{answer_tag}>
            """

            ##### 提示元素 10：预填充 Claude 的回复（如果有）
            PREFILL = f"<{answer_tag}>"

            PROMPT = prompt_combined(ROLE_CONTEXT, TASK_CONTEXT, TONE_CONTEXT, INPUT_DATA, EXAMPLES, TASK_DESCRIPTION,
                                     IMMEDIATE_TASK, PRECOGNITION, OUTPUT_FORMATTING)

            return PROMPT, PREFILL

        prompt, prefill = prompt_creation()
        response_message = self.ai_model_answer_management(query, prompt, prefill, return_all_message,
                                                      self.ai_model, answer_tag=answer_tag, add_ids=True,
                                                      add_response_to_ui=add_response_to_ui)
        return f"{prefill}{response_message}"
    def assistant_with_search_tool(self, query="", role_context="", task_context="", tone_context="", input_data="",
                                   examples="", task_description="", immediate_task="", precognition="",
                                   output_formatting="", pinecone_index="", answer_tag="", threshold=0.5,
                                   top_k=100, return_all_message=False, filename=None, add_response_to_ui=True, callback=None):
        ######################################## 提示元素 ########################################
        def prompt_creation(search_data):
            ##### 提示元素 1：`user` 角色
            ROLE_CONTEXT = f"""{role_context}"""

            ##### 提示元素 2：任务上下文
            TASK_CONTEXT = f"""{task_context}"""

            ##### 提示元素 3：语气上下文
            TONE_CONTEXT = f"""{tone_context}"""

            ##### 提示元素 4：要处理的输入数据
            INPUT_DATA = f"""
            <input_data>
            {input_data}
            <search_data>
            {search_data}
            </search_data>
            </input_data>
            """

            ##### 提示元素 5：示例
            EXAMPLES = f"""{examples}"""

            ##### 提示元素 6：详细的任务描述和规则
            TASK_DESCRIPTION = f"""<query>{query}</query>{task_description}”"""

            ##### 提示元素 7：立即任务描述或请求
            IMMEDIATE_TASK = f"""{immediate_task}"""

            ##### 提示元素 8：预想（逐步思考）
            PRECOGNITION = f"""{precognition}"""

            ##### 提示元素 9：输出格式
            OUTPUT_FORMATTING = f"""
            <{answer_tag}>
            {output_formatting}
            </{answer_tag}>
            """

            ##### 提示元素 10：预填充 Claude 的回复（如果有）
            PREFILL = "" if answer_tag == "" else f"<{answer_tag}>"


            PROMPT = prompt_combined(ROLE_CONTEXT, TASK_CONTEXT, TONE_CONTEXT, INPUT_DATA, EXAMPLES, TASK_DESCRIPTION,
                                     IMMEDIATE_TASK, PRECOGNITION, OUTPUT_FORMATTING)

            return PROMPT, PREFILL

        ######################################## 資料搜索 ########################################

        if pinecone_index != "":
            response_message = self.assistant_normal(
                query=f"分析使用者的問題：「{query}」，從 <input_data> 中提取與問題相關的內容，並放置在 <reference> 標籤中。如果資料不足以回答問題，則進一步判斷是否需要到 Pinecone 資料庫檢索，並將精準的搜尋語句放置於 <search_word> 中；若判定無需檢索，則標記「略過搜索」。",
                role_context="你是一位專精於資料檢索與語意分析的助手，能夠根據用戶問題的性質靈活決定是否需要進行資料搜索。",
                task_context=f"""
                                    用戶向 FreeAger AI 提問，FreeAger 是幫助長者突破年齡限制並探索人生意義的助手。
                                    你的主要任務是幫助 FreeAger AI 整理相關資料和判斷若需進階搜索的精確資料檢索詞語：
                                    1. 分析用戶問題，判斷其核心需求是否與知識檢索相關。
                                    2. 在 <input_data> 中檢索與問題相關的資訊。
                                    3. 判斷檢索資料是否足以回答問題：
                                       - 若足夠，直接生成回答，並標記 <search_word> 為「略過搜索」。
                                       - 若不足，進一步判斷是否需要到 Pinecone 資料庫檢索：
                                         a. 若需要，生成適合語意向量搜索的精準語句，放置於 <search_word>。
                                         b. 若不需要（如問題不涉及檢索或只是簡單互動），標記 <search_word> 為「略過搜索」。
                                """,
                tone_context="請保持專業且簡潔的語氣，根據用戶問題靈活調整回答。",
                input_data=f"",
                examples="",
                task_description=f"""
                                    回答用戶問題時請按照以下步驟：
                                    1. 分析問題意圖，思考要回答這個問題需要那些資料參考，將需要的資料方向列點於 <required_information> 標籤內。
                                    2. 依照問題需求在 <input_data> 中檢索相關資訊，盡可能將相關的結果都放置於 <reference> 標籤內。
                                    3. 判斷 <reference> 中的資訊是否足夠完整回答問題：
                                       - 若足夠，直接生成答案，並於 <search_word> 中標記「略過搜索」。
                                       - 若不足，根據問題性質進行判斷：
                                         a. 如果需要檢索更多資料，生成適合 Pinecone 的搜尋語句並放置於 <search_word>。Pinecone資料庫使用語義搜索，因此搜尋語句使用完整的句子，而不是關鍵字類型。搜尋語句中的學術關鍵字請用中文搭配(英文)來呈現，例如：玩心(playfullness)。搜尋特定關鍵字學術定義時，語句中不用特意加上freeager模型圖，會導致搜尋向量失焦。
                                         b. 如果問題本質不需要檢索，例如用戶問候或互動問題，標記 <search_word> 為「略過搜索」。
                                    4. 你只需回答 <reference> 和 <search_word> 標籤中的內容，不需要回答其他的解釋和說明。     
                                """,
                immediate_task="分析問題並檢索相關資料，根據需要生成搜尋語句或標記略過搜索。",
                precognition="",
                output_formatting=f"""
                                    <reference>...</reference>
                                    <search_word>...</search_word>
                                """,
                answer_tag=f"preprocess_{query}",
                return_all_message=True,
                add_response_to_ui=False
            )
        else:
            response_message = "<reference>略過搜索</reference>"

        preprocess_reference = extract_tagged_content_from_str(response_message, "reference")[0]
        if "略過搜索" in response_message:
            print("此題不需要搜索，略過pinecone搜索")
            search_box = ["無搜尋參考資料，請略過"]
        else:
            picone_search_topic = extract_tagged_content_from_str(response_message, "search_word")[0]
            print(f"picone_search_topic:{picone_search_topic}")
            search_box = self.search_tool(f"{picone_search_topic}", picone_index=pinecone_index, token_budget=7000,
                                     top_k=top_k, filename=filename, threshold=threshold)
        # 如果 search_tool 返回空列表，提供默認值
        if not search_box:
            search_box = ["沒有超過閥值以上的搜尋結果"]

        print(f"=== search_box 內有 {len(search_box)} 筆資料 ===")
        progress = 0
        final_data = ""
        for search_data in search_box:
            if search_data in ["無搜尋參考資料，請略過", "沒有超過閥值以上的搜尋結果"]:
                break
            progress = progress + 1
            print(f"=== 目前正在進行 search_box 中的第  {progress}/{len(search_box)} 筆資料 ===")
            self.assistant_normal(query=f"提取出與「{query}」相關的引文",
                             role_context="你是一位能從大量資料中整理和分析與主題相關資料的AI機器人，專注於提供相關的分析與解釋。",
                             task_context=f"依據 <input_data> 標籤中的輸入資料為來源，將與 「{query}」 相關的引文內容逐步列出。",
                             tone_context="",
                             input_data=f"{search_data}",
                             examples=f"""
                                將你於 <input_data> 中分析出所有相關引言放在 <search_result_file> 標籤中。另外，將原始引文放在 <original> 標籤，並將其繁體中文的說明放在 <explain> 標籤。

                                <example>
                                 <search_result_file = [33-178] goodstein1978.pdf>
                                 <original>
                                 The definition of self-esteem is a person's overall evaluation and perception of their own worth.(Brown, 1993, 1998; Brown & Dutton, 1995)
                                 </original>
                                 <explain>
                                 自尊的定義是一個人對自己價值的整體評價與看法。 (Brown, 1993, 1998; Brown & Dutton, 1995)
                                 </explain>
                                 </search_result_file>
                                </example>
                                """,
                             task_description=f"""回答問題時請依照下列步驟進行：
                                1. 根據 「{query}」 的主題，對 <input_data> 中的資料進行分析，找出相關的引言內容。
                                2. 將原始引言放置在 <original> 標籤中。
                                3. 將原始引言的繁體中文解釋版本放置在 <explain> 標籤中。
                                4. 將 <original> 和 <explain> 的成果，放置在 <search_result_file> 標籤中。
                                5. 僅限於提取重要引文解釋，避免添加無關的內容。""",
                             immediate_task="",
                             precognition="",
                             output_formatting=f"""
                                <search_result_file = [33-178] goodstein1978.pdf>
                                <original>
                                The definition of self-esteem is a person's overall evaluation and perception of their own worth.(Brown, 1993, 1998; Brown & Dutton, 1995)
                                </original>
                                <explain>
                                自尊的定義是一個人對自己價值的整體評價與看法。 (Brown, 1993, 1998; Brown & Dutton, 1995)
                                </explain>
                                </search_result_file>

                                <search_result_file = [70-5] Article Self-Concept by Saul McLeod.pdf-2>
                                <original>
                                Self-esteem refers to the extent to which we like, accept, or approve of ourselves, or how much wevalue ourselves. Self-esteem always involves a degree of evaluation and we may have either a positive
                                or a negative view of ourselves.
                                </original>
                                <explain>
                                自尊是指我們喜歡、接受或認可自己的程度，或者是我們對自己價值的評估。自尊總是涉及某種程度的評價，我們對自己可能持有正面的或負面的看法。
                                </explain>
                                </search_result_file>""",
                             answer_tag=f"data_{query}",
                             return_all_message=True,
                             add_response_to_ui=False)
            final_data = extract_tagged_content_from_messages(self.messages_history_ai, f"data_{query}")
        ######################################## AI提問 ########################################
        prompt, prefill = prompt_creation(f"{preprocess_reference}\n{final_data}")
        response_message = self.ai_model_answer_management(query, prompt, prefill, return_all_message,
                                                      self.ai_model,
                                                      answer_tag=answer_tag, add_ids=True,
                                                      add_response_to_ui=add_response_to_ui)
        if callback:
            callback(prefill, response_message)
        return f"{prefill}{response_message}"

    def ai_model_answer_management(self, message_for_ui, message_for_ai, prefill, return_all_message, ai_model,
                                   temperature=0, answer_tag="", add_ids=False, add_response_to_ui=True):
        if self.print_prompt == True:
            print(f"""
            =======================================================================================================================
            此次的PROMPT:
            {message_for_ai}
            =======================================================================================================================
            """)
        # 判斷這次要丟給AI處理的對話
        messages_for_AImodel = []
        # 判斷是否要加上訓練用對話
        if self.training_message:
            messages_for_AImodel.extend(self.training_message)
        # 判斷是否要加上全部訊息全部訊息
        if return_all_message == True:
            messages_for_AImodel.extend(self.messages_history_ai)
        # 加上此次要給的訊息
        messages_for_AImodel.append({"role": "user", "content": message_for_ai})
        if prefill != "":
            messages_for_AImodel.append({"role": "assistant", "content": prefill})

        # 回答問題
        if "claude" in ai_model.lower():
            response = self.clientCLAUDE.messages.create(
                model=ai_model,
                max_tokens=8192,
                temperature=temperature,
                system="",
                messages=messages_for_AImodel
            )
            response_message = response.content[0].text
        elif ai_model == 'grok-vision-beta':

            response = self.clientXAI.messages.create(
                model=ai_model,
                max_tokens=8192,
                temperature=temperature,
                messages=messages_for_AImodel
            )
            response_message = response.content[0].text
        else:
            if ai_model == 'o1-preview':
                temperature = 1
            response = self.clientGPT.chat.completions.create(
                model=ai_model,
                messages=messages_for_AImodel,
                temperature=temperature
            )
            response_message = response.choices[0].message.content

        print('🤖 ========= AI model 回覆訊息 =====================')
        if add_ids == True:
            response_message = add_reference_ids(response_message)
        print(f"{prefill}\n{response_message}")
        time.sleep(30)

        # ========管理對話紀錄=============
        if answer_tag != "" and extract_tagged_content_from_str(response_message, answer_tag) == "":
            assistant_content = f"<{answer_tag}>\n{prefill}{response_message}\n</{answer_tag}>"
        else:
            assistant_content = f"{prefill}{response_message}"
        # 管理給ai的對話紀錄
        self.messages_history_ai.append({"role": "user", "content": message_for_ui})
        self.messages_history_ai.append({"role": "assistant", "content": assistant_content})
        # 管理給ui的對話紀錄
        if add_response_to_ui == True:
            self.messages_history_ui.append({"role": "user", "content": message_for_ui})
            self.messages_history_ui.append({"role": "assistant", "content": assistant_content})
        time.sleep(1)
        return response_message
    def search_tool(self, query: str, token_budget: int, top_k=10, picone_index="freeager-scholar-middle-chunk",
                    filename=None, threshold=0.5) -> list:
        """Return a list of message strings, with relevant source texts pulled from a dataframe."""

        # Query articles using the provided query
        strings, relatednesses, indices, filenames, sections = self.pinecone_query_article(query, picone_index=picone_index,
                                                                                      filename=filename, top_k=top_k,
                                                                                      threshold=threshold)

        # 檢查是否有任何相關性達到標準的結果
        if not strings or strings == ['N/A']:
            print('No matches above the threshold.')
            return []

        # Create an empty list to store messages
        messages = []
        current_message = ""

        # 創建空的 DataFrame
        df = pd.DataFrame(columns=['Relatedness', 'FileName-section', 'Text'])

        # Process each result and add to the message string
        for string, relatedness, index, filename, section in zip(strings, relatednesses, indices, filenames, sections):
            next_article = f'\n<search_result_file = {filename}-{section}>\n{string}\n</search_result>\n'

            # If adding the next article exceeds the token limit, finalize the current message and start a new one
            if num_tokens(current_message + next_article, model="gpt-3.5-turbo") > token_budget:
                # Append the current message to the messages list
                messages.append(current_message)
                # Reset current_message to start a new one
                current_message = next_article
            else:
                # If it doesn't exceed the token budget, keep adding to the current message
                current_message += next_article

            # Add data to the DataFrame for reference table
            new_row = {
                'Relatedness': relatedness,
                'FileName-section': f"{filename}-{section}",
                'Text': string.replace('\n', ' ')
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Append the final message if there's remaining content
        if current_message:
            messages.append(current_message)

        # 顯示結果的DataFrame表格
        print(f"search_result_df:{df}")
        # Return the list of message strings
        return messages
    def pinecone_query_article(self, query, picone_index="freeager-scholar-middle-chunk", filename=None, top_k=5,
                               threshold=0.6):

        '''Queries an article using the provided query and prints results.'''
        print('=== 至 pinecone 搜尋資料中 ===')

        # Create vector embeddings based on the query
        query_embedding_response = self.clientGPT.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        embedded_query = query_embedding_response.data[0].embedding
        # 檢查是否有指定 filename 並設置篩選條件
        filter_condition = {}
        if filename:
            filter_condition = {'filename': {'$eq': filename}}

        index_name = picone_index
        print(f"index_name:{index_name}")

        index = self.pc.Index(name=index_name)
        # Query the index using the query vector and apply filter by filename
        query_result = index.query(
            vector=embedded_query,
            namespace='content',
            top_k=top_k,
            filter=filter_condition  # 使用 filter 參數來根據 filename 篩選
        )

        # Print query results
        if not query_result.matches:
            print('No query result')
            return

        matches = query_result.matches
        indices = [str(res.id) for res in matches]
        relatednesses = [res.score for res in matches]
        # Fetch metadata for the matched vectors
        response = index.fetch(ids=indices, namespace='content')

        strings, sections, filenames = [], [], []
        for vector_id in indices:
            if 'vectors' in response and vector_id in response['vectors']:
                vector_info = response['vectors'][vector_id]
                if 'metadata' in vector_info:
                    metadata = vector_info['metadata']
                    strings.append(metadata.get('text', 'N/A'))
                    sections.append(metadata.get('section', 'N/A'))
                    filenames.append(metadata.get('filename', 'N/A'))
                else:
                    print(f"No metadata found for vector ID {vector_id}")
                    strings.append('N/A')
                    sections.append('N/A')
                    filenames.append('N/A')
            else:
                print(f"No vector found for ID {vector_id} in namespace {'content'}")
                strings.append('N/A')
                sections.append('N/A')
                filenames.append('N/A')

        df = pd.DataFrame({
            'indices': indices,
            'relatednesses': relatednesses,
            'strings': strings,
            'filenames': filenames,
            'sections': sections
        })

        # 过滤掉相关性小于阈值的行
        df = df[df['relatednesses'] >= threshold]

        # 打印过滤后的DataFrame
        # print(df.head(30))

        # 如果没有任何结果超出阈值，返回空列表
        if df.empty:
            print('No matches above the threshold.')
            return [], [], [], [], []
        # Return all results in the desired format
        strings = df['strings'].tolist()
        relatednesses = df['relatednesses'].tolist()
        indices = df['indices'].tolist()
        filenames = df['filenames'].tolist()
        sections = df['sections'].tolist()

        return strings, relatednesses, indices, filenames, sections

    def save_message_to_csv(self):
        # 將 extended_messages_history 轉換為 DataFrame 並打印
        messages_history_df_ai = pd.DataFrame(self.messages_history_ai)
        messages_history_df_ui = pd.DataFrame(self.messages_history_ui)

        current_time_str = datetime.now().strftime("%Y%m%d")

        # 如果 DataFrame 不為空才進行存檔
        if not messages_history_df_ai.empty:
            message_file_path_ai = f"{self.message_folder_path}{self.bot_name}_{current_time_str}_ai.csv"
            self._append_to_csv(messages_history_df_ai, message_file_path_ai)
            print(f"messages_history_ai 對話紀錄已存檔至 {message_file_path_ai}")
        else:
            print("messages_history_ai 對話紀錄是空的，未進行存檔。")

        # 如果 DataFrame 不為空才進行存檔
        if not messages_history_df_ui.empty:
            message_file_path_ui = f"{self.message_folder_path}{self.bot_name}_{current_time_str}_ui.csv"
            self._append_to_csv(messages_history_df_ui, message_file_path_ui)
            print(f"messages_history_ui 對話紀錄已存檔至 {message_file_path_ui}")
        else:
            print("messages_history_ui 對話紀錄是空的，未進行存檔。")

    # ✅ 新增這個輔助函式（僅附加寫入）
    def _append_to_csv(self, df, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        write_header = not os.path.exists(file_path)  # 檔案不存在時才寫入表頭
        df.to_csv(file_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
        print(f"✅ 已附加 {len(df)} 筆對話至：{file_path}")
    def clear_all_messages(self):
        self.messages_history_ai = []
        self.messages_history_ui = []


def prompt_combined(ROLE_CONTEXT, TASK_CONTEXT, TONE_CONTEXT, INPUT_DATA, EXAMPLES, TASK_DESCRIPTION, IMMEDIATE_TASK,
                    PRECOGNITION, OUTPUT_FORMATTING):
    PROMPT = ""

    if ROLE_CONTEXT:
        PROMPT += f"""{ROLE_CONTEXT}"""

    if TASK_CONTEXT:
        PROMPT += f"""\n\n{TASK_CONTEXT}"""

    if TONE_CONTEXT:
        PROMPT += f"""\n\n{TONE_CONTEXT}"""

    if INPUT_DATA:
        PROMPT += f"""\n\n{INPUT_DATA}"""

    if EXAMPLES:
        PROMPT += f"""\n\n{EXAMPLES}"""

    if TASK_DESCRIPTION:
        PROMPT += f"""\n\n{TASK_DESCRIPTION}"""

    if IMMEDIATE_TASK:
        PROMPT += f"""\n\n{IMMEDIATE_TASK}"""

    if PRECOGNITION:
        PROMPT += f"""\n\n{PRECOGNITION}"""

    if OUTPUT_FORMATTING:
        PROMPT += f"""\n\n{OUTPUT_FORMATTING}"""


    return PROMPT
def get_filepath_to_messages(message_file_path):
    if os.path.exists(message_file_path):
        # 讀取 CSV 檔案並轉換成對應的格式
        messages_history_df = pd.read_csv(message_file_path)
        messages = messages_history_df.to_dict(orient='records')
    # 檢查最後一筆資料的 role，如果是 'user'，則刪除
    if messages and messages[-1]["role"] == "user":
        messages.pop()
    return messages
def extract_tagged_content_from_str(input_str, tag_name):
    """
    提取指定標籤中的內容，並清理內容中的多餘空白。

    參數:
    input_str (str): 包含一個或多個特定標籤的字符串。
    tag_name (str): 需要提取內容的標籤名稱。

    返回:
    List[str]: 包含所有提取出來並清理過的標籤內容的列表。
    """
    pattern = rf'<{tag_name}\s*[^>]*>(.*?)</{tag_name}>'
    # 使用 re.DOTALL 來跨行匹配
    contents = re.findall(pattern, input_str, re.DOTALL)
    # 清理提取出的內容
    cleaned_contents = [' '.join(content.split()) for content in contents]
    return cleaned_contents
def extract_tagged_content_from_messages(messages, tag, only_extract_first=False):
    """
    提取對話紀錄中，指定標籤(tag)的所有內容。

    :param messages: 包含多個訊息的列表，每個訊息是字典形式
    :param tag: 要提取內容的標籤名稱
    :param only_extract_first: 是否僅提取第一個標籤的內容，默認為 False
    :return: 包含所有提取內容的字串
    """
    message_tagged_str = ""
    pattern = f"<{tag}>(.*?)</{tag}>"

    for message in messages:
        content = message.get('content', '')
        if isinstance(content, str):
            # 使用正則表達式找到所有符合的標籤內容
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                if only_extract_first:
                    # 如果只提取第一個，加入第一個匹配並停止
                    message_tagged_str += matches[0].strip() + "\n\n"
                    return message_tagged_str  # 提取第一個後立即返回
                else:
                    # 否則提取所有匹配內容
                    for match in matches:
                        message_tagged_str += match.strip() + "\n\n"

    return message_tagged_str
def add_reference_ids(text):
    global global_reference_id  # 使用全域變數

    # 找到所有 <search_result_file = XXXX>
    pattern = r"<search_result_file = (.+?)>"
    matches = re.findall(pattern, text)

    # 替換每個匹配的項目，加上 reference_id 編號
    for match in matches:
        new_tag = f"<search_result_file = reference_id_{global_reference_id}: {match}>"
        text = text.replace(f"<search_result_file = {match}>", new_tag, 1)
        global_reference_id += 1  # 每次替換後遞增全域編號

    return text
def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
def save_message_to_csv(messages_history, message_file_path):
    ensure_directory_exists(message_file_path)
    # 將 extended_messages_history 轉換為 DataFrame 並打印
    messages_history_df = pd.DataFrame(messages_history)
    # 如果 DataFrame 不為空才進行存檔
    if not messages_history_df.empty:
        messages_history_df.to_csv(message_file_path, index=False)
        print(f"對話紀錄已存檔至 {message_file_path}")
    else:
        print("DataFrame 是空的，未進行存檔。")
    print(f"""=== 對話歷史紀錄 === \n{messages_history_df}""")
def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':

    # 創建一個 GPT-4o 機器人
    chatbot1 = AIChatbot(bot_name="BotA", ai_model="gpt-4o")
    chatbot1.read_previous_messages("./message_record/20241205.csv")
    # 創建一個 Claude-3.5 機器人
    chatbot2 = AIChatbot(bot_name="BotB", ai_model="claude-3-7-sonnet-20250219")

    print("BotA 對話紀錄:", chatbot1.get_messages_history())
    print("BotB 對話紀錄:", chatbot2.get_messages_history())
    response2 = """
    <prompt>
    根據以下對話內容，分析其中隱含的核心信念、情緒狀態，以及對話者的可能人生故事。請依照以下格式輸出：

    1. 隱含信念系統：
       - 核心價值觀和世界觀：對話者如何看待世界以及他們認為重要的價值
       - 自我認知和身份認同：對話者如何看待自己以及他們的角色和身份
       - 對人際關係的基本假設：對話者與他人互動的基本信念和期待
    
    2. 情緒全景：
       - 主導情緒及其強度：目前主導對話者的情緒及其強烈程度
       - 情緒衝突或矛盾：對話者在情緒上可能存在的內在衝突或矛盾
       - 情緒變化的關鍵觸發點和應對模式：哪些事件或話語引發情緒變化，對話者如何應對
    
    3. 生命敘事重構：
       - 推測關鍵生活經歷和轉折點：可能塑造對話者信念和行為的重大事件或時刻
       - 形成的行為模式和思維習慣：由這些經歷所產生的行為和思考方式
       - 這些經歷如何形塑當前的行為和決策：過去的經歷如何影響對話者現在的選擇和行動
    
    4. 分析證據：
       - 提供具體的對話內容或語句作為支持分析的證據，以增加分析的可信度和實用性
    </prompt>
    你是一位prompt工程師，我想從對話中，分析對話中隱藏的信念、情緒和人生故事。我的prompt該怎麼設計？請幫我優化
    將優化過後的prompt放置於<better_prompt>中。
    """
    for i in range(5):
        print("=============================")
        print(f"第 {i} 輪對話")
        print("=============================")
        # 發送訊息
        response1 = chatbot1.send_message(f"{response2}\n我想從對話中，分析對話中隱藏的信念、情緒和人生故事。我的prompt該怎麼設計？請幫我優化")
        print("BotA 回應:\n", response1)
        response2 = chatbot2.send_message(f"{response1}\n我想從對話中，分析對話中隱藏的信念、情緒和人生故事。我的prompt該怎麼設計？請幫我優化")
        print("BotB 回應:\n", response2)


    # 獲取對話歷史
    print("BotA 對話紀錄:", chatbot1.get_messages_history())
    print("BotB 對話紀錄:", chatbot2.get_messages_history())

"""
1. 隱含信念系統：
   - 核心價值觀和世界觀
   - 自我認知和身份認同
   - 對人際關係的基本假設

2. 情緒全景：
   - 主導情緒及其強度
   - 情緒衝突或矛盾
   - 情緒變化的關鍵觸發點和應對模式

3. 生命敘事重構：
   - 推測關鍵生活經歷和轉折點
   - 形成的行為模式和思維習慣
   - 這些經歷如何形塑當前的行為和決策



"""