---
layout: post
title: tiktoken 및 cl100k_base을 오프라인에서 사용후기
---

가끔 프로젝트를 하다보면 tiktoken을 오프라인으로 사용해야될 경우가 필요하다.

주로 캐시파일에 저장되기 때문에 시간이 지난다면 tiktoken이 알아서 새롭게 다운 받으려고 하는데... 이 경우 오프라인 PC에서 사용하게 되거나 제한된 인터넷 환경에서는 좀 귀찬게 된다. 


**첫번째. tiktoken 파일 다운로드**

```
import tiktoken_ext.openai_public
import inspect

print(dir(tiktoken_ext.openai_public))
print(inspect.getsource(tiktoken_ext.openai_public.cl100k_base))

# >>> 
# 이하 생략
def cl100k_base():
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
```

mergeable\_ranks 속에 있는 링크를 얻게 된다. tiktoken에서 사용될 토크나이저 파일을 다운 받는 URL을 얻을 수 있다.

**두번째. 캐시 파일 이름 가져오기**

blob를 통해서 아까 mergeable\_ranks 속에 있는 url을 blob path로 사용해서 캐시 키를 얻습니다. 캐시키값을 아까 다운 받은 파일의 이름으로 변경합니다.

```
import hashlib

blobpath = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
print(cache_key)

>>>9b5ad71b2ce5302211f9c61530b329a4922fc6a4
```

**세번째. tiktoken 캐시 설정**

마지막으로 원하는 위치에 환경변수로 TIKTOKEN\_CACHE\_DIR을 설정해준다면 이제 계속 사용할 수 있는 상태가 됩니다.

```
import os

tiktoken_cache_dir = "path_to_folder_containing_tiktoken_file"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

encoding = tiktoken.get_encoding("cl100k_base")
encoding.encode("안녕하세요 여러분!!!")
```

정말 필요한 자료를 여기서 얻어서 한글화 시켜서 저장합니다. 감사합니다.

↓ ↓ ↓ ↓ ↓

[https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer](https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer)
