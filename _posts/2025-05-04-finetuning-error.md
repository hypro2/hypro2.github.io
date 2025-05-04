---
layout: post
title: finetuning error 모음
---
RuntimeError: Cannot find valid samples, check data/README.md for the data format

시스템, 어시스턴트, 유저 순서 에러, 어시스턴트가 유저보다 절대 먼저 나올 없음.



AttributeError: 'NoneType' object has no attribute 'cdequantize_blockwise_fp32'

언슬로스 설치 오류 언슬로스 먼저 정상 설치 확인 후 라마팩토리 설치



AssertionError: Backwards requires embeddings to be bf16 or fp16

믹스드 프리시전 훈련 오류 pure bf16으로 대처



ValueError: The number of audios does not match the number of <audio> tokens in

content에 <audio>,<video>,<image>가 있으면 에러남 텍스트에서 제거



raise DatasetGenerationError("An error occurred while generating the dataset") from e datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset

데이터셋 너무 큰 크기로 인한 에러 1기가정도로 분할 조치
