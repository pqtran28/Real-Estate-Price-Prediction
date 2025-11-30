from google import genai
from google.genai import types
from google.genai import errors
from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
import time
from dotenv import load_dotenv
import os
import logging
from sqlalchemy import create_engine

MODEL_ID = 'gemini-2.0-flash'
LIMITS = {
    'req_per_min': 15,
    'req_per_day' : 1000,
    'token_per_min': 1000000
}
MISS_LINE = []
INPUT_EXCEED_LINE = []
SQL_QUERY = "select * from raw_data"

def load_environment_variables():

    load_dotenv()

    config_dict = {
        'host': os.getenv("PG_HOST"),
        'port': os.getenv("PG_PORT"),
        'database': os.getenv("PG_DATABASE"),
        'user': os.getenv("PG_USER"),
        'password': os.getenv("PG_PASSWORD"),
        'api_key' : os.getenv("api_key")
    }
    
    return config_dict

def connect_to_database(config: dict):
    try:
        connection_url = (
            f"postgresql+psycopg2://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )
        engine = create_engine(connection_url)
        with engine.connect() as _:
            print("✅ Connected.")
        return engine
    except Exception as e:
        print("❌ Failed.", e)
        return None

def load_data(connection):
    if connection:
        data = pd.read_sql(
            SQL_QUERY,
            con=connection
        )
        if data.empty:
            print("❌ Load data failed.")
            return None
        else:
            print("✅ Load data successfully.")
            return data

class DescriptionInfo(BaseModel):
    loai_hinh_giaodich: Optional[Literal["bán", "cho thuê"]]
    noi_that: Optional[Literal["hạng sang", "cao cấp", "đầy đủ", "trung bình", "cơ bản", "không có nội thất"]]
    tien_ich: Optional[str]
    huong_nha_dat: Optional[Literal[
        "Bắc",
        "Nam",
        "Đông",
        "Tây",
        "Đông Bắc",
        "Tây Bắc",
        "Đông Nam",
        "Tây Nam"
    ]]
    phap_ly: Optional[Literal[
        "sổ đỏ",
        "sổ hồng",
        "sổ riêng",
        "đã hoàn công",
        "sổ chung",
        "giấy tay",
        "đang chờ sổ",
        "chưa tách thửa",
        "không giấy tờ",
        "vướng quy hoạch",
        "đất quy hoạch",
        "đất nông nghiệp chưa chuyển mục đích"
    ]]
    khu: Optional[Literal[
        "đô thị mới",
        "ven đô thị",
        "trung tâm thành phố/quận",
        "giáp ranh trung tâm thành phố/quận",
        "đông dân cư",
        "thưa thớt dân cư",
        "vùng ngoại thành",
        "vùng sâu vùng xa",
        "gần/trong khu du lịch/nghỉ dưỡng",
        "gần/ trong khu công nghiệp",
        "ven sông",
        "gần/trong khu nhà trọ"
    ]]
    do_thi_hoa: Optional[Literal[
        "trung tâm đô thị",
        "khu dân cư hiện hữu ổn định",
        "đô thị đang phát triển",
        "hạ tầng chưa hoàn thiện",
        "bán đô thị",
        "nông thôn"
    ]]
    tinh_trang: Optional[Literal[
        "đã xây",
        "đang xây",
        "đất trống",
        "cần sửa chữa"
    ]]
    an_ninh: Optional[Literal[
        "được đảm bảo (có bảo vệ và camera)",
        "trung bình (có camera hoặc bảo vệ)"
    ]]
    vi_tri: Optional[Literal[
        "quốc lộ",
        "trục đường chính",
        "đường lớn",
        "ngã tư",
        "bùng binh",
        "hẻm vừa",
        "hẻm xe hơi",
        "hẻm nhỏ"
    ]]


def call_llm_API(des,client,index: int,input_lim: int):
    logger = logging.getLogger('API_logger')
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    response = None
    try:
        total_tokens = client.models.count_tokens(
             model=MODEL_ID, 
             contents=f'Hãy rút trích, đánh giá thông tin bất động sản từ mô tả {des}.Trả về **đầy đủ các trường trong JSON**,**ngay cả khi không có thông tin** (khi đó gán giá trị là None). Không được bỏ bất kỳ field nào.'
             # prompt
            #  Hãy rút trích thông tin HƯỚNG CHÍNH (direction) của bất động sản từ mô tả {des}.Trả về **đầy đủ các trường trong JSON**,**ngay cả khi không có thông tin** (khi đó gán giá trị là None). Không được bỏ bất kỳ field nào.
            ).total_tokens
        if total_tokens > input_lim:
            INPUT_EXCEED_LINE.append(index)
            return response
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=f'Hãy rút trích, đánh giá thông tin bất động sản từ mô tả {des}.Trả về **đầy đủ các trường trong JSON**,**ngay cả khi không có thông tin** (khi đó gán giá trị là None). Không được bỏ bất kỳ field nào.',
             # prompt
            #  Hãy rút trích thông tin HƯỚNG CHÍNH (direction) của bất động sản từ mô tả {des}.Trả về **đầy đủ các trường trong JSON**,**ngay cả khi không có thông tin** (khi đó gán giá trị là None). Không được bỏ bất kỳ field nào.
            config=types.GenerateContentConfig(
                system_instruction='You are a realtor who works in real estate market in Vietnam.',
                response_mime_type= 'application/json',
                response_schema= DescriptionInfo,
                temperature= 0.2
            )
        )
        print(f'✅ Response generated. The program is still running, line {index}')
    except errors.APIError as e:
        logger.error(f'An error occurs. Error code: {e.code}, {e.status}, line {index}')
        logger.warning('Response will be None.')
        logger.info('The program is still running.')
        if (e.code == 429):
            time.sleep(20)
        MISS_LINE.append((e.code,index))
    except Exception as e:
        logger.error(f'An unknown error occurs: {e}, line {index}')
        MISS_LINE.append(('unknown',index))
    return response

def fetch_data_mota(df: pd.DataFrame, api_keys: list[str], col_to_extract: str):

    used_key = []
    count_extracted = 0
    stop_at = 0
    for key in api_keys:
        client = genai.Client(api_key=key)
        model_info = client.models.get(model=MODEL_ID)
        input_limit = model_info.input_token_limit
        start = time.time()
        tpm = 0
        rpm = 0
        for i in df.index[df.index.get_loc(stop_at):]:
            if pd.isna(df.iloc[i][col_to_extract]) or (df.iloc[i][col_to_extract] is None):
                continue
            response = call_llm_API(des= df.at[i,col_to_extract],client=client,index=i,input_lim=input_limit)
            time.sleep(4)
            if response is not None:
                df.at[i,'extracted'] = response.text
                count_extracted += 1
                if (count_extracted > LIMITS['req_per_day']):
                    stop_at = i + 1
                    break
                tpm += response.usage_metadata.total_token_count
                rpm += 1
                end = time.time()
                if(start + 60 >= end) and (tpm >= LIMITS['token_per_min'] or rpm >= LIMITS['req_per_min']):
                    time.sleep(60 - (end - start))
                    tpm = 0
                    rpm = 0
                    start = time.time()
        used_key.append(key)
    print(used_key)
    return df, stop_at

config = load_environment_variables()
connection = connect_to_database(config=config)
api_key_0 = config['api_key']
# data = load_data(connection=connection)
data = pd.read_csv('raw_data.csv')
data20_40, stop = fetch_data_mota(data.iloc[0:20],api_keys=[api_key_0],col_to_extract='mo_ta')
data20_40.to_csv('test_extracted.csv',index=False)
print(stop)
print(MISS_LINE)





