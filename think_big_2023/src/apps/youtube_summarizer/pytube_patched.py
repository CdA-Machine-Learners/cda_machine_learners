from typing import List
from html import unescape
from pytube import YouTube, request, Caption
import xml.etree.ElementTree as ElementTree
import json


class PatchedCaption(Caption):
    @property
    def json_captions(self) -> dict:
        # bug fix, json wasn't imported in main file
        """Download and parse the json caption tracks."""
        json_captions_url = self.url.replace('fmt=srv3', 'fmt=json3')
        text = request.get(json_captions_url)
        parsed = json.loads(text)
        assert parsed['wireMagic'] == 'pb3', 'Unexpected captions format'
        return parsed

    @property
    def scc_captions(self) -> str:
        # Added SCC Support
        """Download and parse the scc caption tracks."""
        scc_captions_url = self.url.replace('fmt=srv3', 'tfmt=scc')
        text = request.get(scc_captions_url)
        return [{'text': unescape(c.text), **c.attrib} for c in ElementTree.fromstring(text)]


class CustomYouTube(YouTube):
    @property
    def chapters(self):
        def time_to_seconds(time_str):
            time_parts = list(map(int, time_str.split(':')))
            if len(time_parts) == 3:
                h, m, s = time_parts
            elif len(time_parts) == 2:
                h = 0
                m, s = time_parts
            else:
                raise ValueError(f"Invalid time format: {time_str}")
            return h * 3600 + m * 60 + s

        def if_tuple_get_first(t):
            if isinstance(t, tuple):
                return t[0]
            return t

        engagement_panels = self.initial_data.get('engagementPanels')
        chapters = []
        for panel in engagement_panels:
            contents = panel.get('engagementPanelSectionListRenderer', {}).get('content', {}).get(
                'macroMarkersListRenderer', {}).get('contents', [])
            for c in contents:
                title = c.get('macroMarkersListItemRenderer', {}).get('title', {}).get('simpleText'),
                timestamp = c.get('macroMarkersListItemRenderer', {}).get('timeDescription', {}).get('simpleText'),
                a11y_label = c.get('macroMarkersListItemRenderer')['timeDescriptionA11yLabel'],
                relative_url = c.get('macroMarkersListItemRenderer').get('onTap', {}).get('commandMetadata', {}).get(
                    'webCommandMetadata', {}).get('url')

                chapter = {
                    'title': if_tuple_get_first(title),
                    'timestamp': if_tuple_get_first(timestamp),
                    'timestamp_seconds': time_to_seconds(if_tuple_get_first(timestamp)),
                    'a11y_label': if_tuple_get_first(a11y_label),
                    'relative_url': if_tuple_get_first(relative_url),
                }
                chapters.append(chapter)
        return chapters

    @property
    def caption_tracks(self) -> List[PatchedCaption]:
        """Get a list of :class:`Caption <Caption>`.

        :rtype: List[Caption]
        """
        raw_tracks = (
            self.vid_info.get("captions", {})
            .get("playerCaptionsTracklistRenderer", {})
            .get("captionTracks", [])
        )
        return [PatchedCaption(track) for track in raw_tracks]
