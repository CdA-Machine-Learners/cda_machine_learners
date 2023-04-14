from time import sleep
from youtube_summarizer.map_reduce_continue import MapReduceContinue
from youtube_summarizer.pytube_patched import CustomYouTube


class YoutubeSummarizer:
    def __init__(self, yt_video, debug=False):
        self.yt_video = yt_video
        self.yt = self._get_youtube_with_retries()
        self.mrc = MapReduceContinue(debug=debug)
        self._formatted_transcripts = None
        self._grouped_transcripts = None
        self.debug = debug

    def _get_youtube_with_retries(self, max_retries=1):
        for i in range(max_retries):
            try:
                yt = CustomYouTube(self.yt_video)
                if not yt.caption_tracks[0]:
                    raise ValueError("No initial data")
                return yt
            except Exception as e:
                print('Error getting Youtube video. Retrying...')
                sleep(i + 2)
        raise ValueError("Could not get Youtube video")

    def _group_transcripts_by_chapter(self):
        chapters = self.yt.chapters
        if not chapters:
            return

        grouped_transcripts = []
        chapter_index = 0
        transcripts = self.yt.caption_tracks[0].scc_captions

        for chapter in chapters:
            current_chapter_start = chapter['timestamp_seconds']

            # Check if we have reached the next chapter
            if chapter_index + 1 < len(chapters):
                next_chapter_start = chapters[chapter_index + 1]['timestamp_seconds']
            else:
                next_chapter_start = float('inf')

            # Group transcripts by chapter
            chapter_transcripts = []
            while transcripts and current_chapter_start <= float(transcripts[0]['start']) < next_chapter_start:
                chapter_transcripts.append(transcripts.pop(0))

            # Append the chapter object to the results
            grouped_transcripts.append({
                'title': chapter['title'],
                'transcripts': chapter_transcripts
            })
            chapter_index += 1

        self._grouped_transcripts = grouped_transcripts

    def _format_transcripts(self):
        buffer = f'# {self.yt.title}\n'
        if self.yt.chapters:
            for chapter in self.grouped_transcripts:
                buffer += f"\n\n{chapter['title']}\n"
                buffer += ' '.join([transcript['text'] for transcript in chapter['transcripts']])
            self._formatted_transcripts = buffer
        else:
            self._formatted_transcripts = ' '.join([c['text'] for c in self.yt.caption_tracks[0].scc_captions])

    @property
    def grouped_transcripts(self):
        if self._grouped_transcripts is None:
            self._group_transcripts_by_chapter()
        return self._grouped_transcripts

    @property
    def formatted_transcripts(self):
        if self._formatted_transcripts is None:
            self._format_transcripts()
        return self._formatted_transcripts

    def chapter_aware_summarize(self):
        if not self.yt.chapters:
            raise ValueError("This video does not have chapters")
        for chapter in self.grouped_transcripts:
            text = f'# {chapter["title"]}\n{" ".join([t["text"] for t in chapter["transcripts"]])}'
            chapter['summary'] = self.mrc.summarize(text, chapter=chapter['title'])
        formatted_summary = f'# {self.yt.title}\n\n'
        for chapter in self.grouped_transcripts:
            formatted_summary += f'## {chapter["title"]}\n{chapter["summary"]}\n*****\n'
        return formatted_summary

    def summarize(self):
        if self.yt.chapters:
            if self.debug:
                print(f"Summarizing by chapter")
            return self.chapter_aware_summarize()
        else:
            if self.debug:
                print(f"Summarizing whole transcript")
            return f'# {self.yt.title}\n\n{self.mrc.summarize(self.formatted_transcripts)}'

    def __repr__(self):
        return f"<YoutubeSummarizer: {self.yt.title}>"
