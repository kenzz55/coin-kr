[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xFhGzYc0)
# 한국 동전 합계 계산

단일 RGB 이미지에 촬영된 한국 주회들의 합계 금액을 구하여 출력하세요.

# 입출력 예시
## 입력 예시
```bash
python main.py --input path/to/image.jpg
```
- input의 인자로 이미지의 파일명(현재 폴더가 아닌 경우 경로 포함)을 명령행 인자로 입력
- 자동 채점 시 script 상에서 project 폴더 밖에 있는 다른 경로에 있는 이미지를 사용할 예정

## 출력 예시
```bash
500:2
100:4
50:3
10:0
1550
```
- 각 주화별로 인식된 동전의 수를 먼저 출력
  - 각 줄에는 아래의 양식을 반드시 지켜서 추가 공백 없이 출력  
  - 종류:[count]
    - 500원 동전 2개가 검출된 경우의 예시) 500:2
  - 검출되지 않았더라도 4가지 주화의 수를 모두 출력
- 금액 합계를 마지막 줄에 출력
  - 다른 추가 문자열/공백 없이 금액만 출력
- **텍스트 출력 외에 cv2.imshow 등으로 이미지를 띄우지 마세요.**
  - 결과 이미지를 띄우고 waitkey 함수 등으로 대기하는 경우 수행 시간이 초과되어 0점 처리 될 수 있습니다.
  - main.py의 예시 코드에 있는 imshow를 비롯한 예시 코드는 지우고 작성해주세요.
- 최초 실행 후 채점 서버에서 1분 이상 소요되는 경우 오답 처리됩니다.

# 과제 시작하기
1. Repository clone 하기 (<your-assignment-repo-url> 부분은 본인의 github repo 주소로 치환)
```bash
git clone <your-assignment-repo-url>
cd <your-assignment-repo>
```

2. (권장사항) Virtual environment 생성 (아래의 .venv 는 가상환경의이름이면서 venv가 위치할 경로) 
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. 필요 패키지 설치
```bash
pip install -r requirements.txt
```
- 채점을 위한 가상 환경에서 원본 requirements.txt로 필요 패키지들을 설치한 후 채점 수행
- 과제에서 requirements.txt를 수정해도 된다고 명시된 경우에만 매번 위의 명령어로 패키지들을 설치 후 채점 수행
- 만약 requirements.txt 수정 불가 과제에서 추가로 필요한 패키지가 있다면 메일로 문의

# 사용 패키지
- NumPy 2.2.6
- opencv-python 4.12.0.88

# 📦 과제 제출하기
1. 코드가 오류 없이 실행되는지 확인하세요.
2. 다양한 입력으로로 테스트해보세요.
3. 모든 요구사항을 만족하는지 검증하세요.
4. 변경사항을 Github에 ***commit***하고 ***push***하세요.
   - ⚠️ push되지 않은 내용은 채점되지 않습니다.
   - ⚠️ commit log는 사후 검증에 활용될 수 있습니다.

# 주의사항
- 채점에는 Python 3.13 버전을 사용합니다.
- main.py에서 *main* 함수를 수정하여 과제를 수행하세요.
- 과제 조건에 따라 *parse_args* 함수를 수정하여 과제를 수행하세요.
- 필요에 따라 새로운 함수나 파일을 추가하셔도 됩니다.
- *sys.exit(main())* 부분은 수정하지 마세요.
- 과제에서 명시하는 경우를 제외하고는 requirements.txt 파일을 수정하지 마세요.

# 채점 기준
- 채점시 repository를 clone 받아서, requirements.txt를 기준으로 패키지들을 설치합니다.
- main.py 파일을 명령행 인자(command-line arguments)와 함께 실행합니다.
- 출력물로 지정된 문자열 혹은 image 외에 다른 출력이 나올 경우 오답 처리 됩니다.
  - 예시) Hello가 정답인 경우
    - ❌ 대소문자 불일치: hello 는 오답 처리
  - 예시) 3이 정답인 경우: 
    - ❌ 불필요한 추가 문자열: (정답은 3) 는 오답 처리
  - 예시) 5,2가 정답인 경우: 
    - ❌ 불필요한 빈 칸: 5, 2 는 오답처리


# 📚 참고자료
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Python Argparse Tutorial](https://docs.python.org/3/library/argparse.html)

# 자주 발생하는 오류
1. **ImportError**: 필요한 패키지가 모두 설치되었는지 확인하세요
   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError**: 입력 파일이 존재하는지, 경로가 올바른지 확인하세요

3. **Permission Error**: 출력 디렉토리에 쓰기 권한이 있는지 확인하세요

4. **Memory Error**: 큰 이미지의 경우, 청크 단위로 처리하거나 이미지 크기를 줄이는 것을 고려하세요
