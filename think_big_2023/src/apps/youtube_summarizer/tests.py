from apps.youtube_summarizer.summarizer import YoutubeSummarizer
import time
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{current_time}.md"


if __name__ == '__main__':
    yt = YoutubeSummarizer('https://www.youtube.com/watch?v=UsWB1XodUKA', debug=True)
    with open(file_name, 'w') as file:
        file.write(yt.summarize())