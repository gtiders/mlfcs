"""Align rendered notebook links and heading anchors with MkDocs navigation."""

from __future__ import annotations

import posixpath
import re
from html import escape, unescape
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit

from mkdocs.utils import get_relative_url


def _heading_text(html: str) -> str:
    """Normalize heading text for matching notebook headings to TOC entries."""
    text = unescape(re.sub(r"<[^>]*>", "", html)).removesuffix("¶")
    return " ".join(text.split())


def _toc_entries(items):
    """Yield heading identifiers and titles in the page's TOC order."""
    for item in items:
        yield item.id, _heading_text(item.title)
        yield from _toc_entries(item.children)


def on_page_content(html, *, page, config, files):
    """Repair notebook HTML without altering notebook sources or outputs."""
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    headings = list(_toc_entries(page.toc))

    def heading(match):
        """Use the TOC identifier for the corresponding rendered heading."""
        opening, body, closing = match.groups()
        title = _heading_text(body)
        for index, (identifier, name) in enumerate(headings):
            if name == title:
                headings.pop(index)
                opening = re.sub(r'\s+id=["\'][^"\']*["\']', "", opening)
                opening = opening[:-1] + f' id="{escape(identifier, quote=True)}">'
                body = re.sub(
                    r'(<a\b[^>]*\bclass=["\']anchor-link["\'][^>]*\bhref=)["\'][^"\']*["\']',
                    lambda anchor, identifier=identifier: (
                        anchor[1] + f'"#{escape(identifier, quote=True)}"'
                    ),
                    body,
                )
                return opening + body + closing
        return match[0]

    html = re.sub(r"(<h[1-6]\b[^>]*>)(.*?)(</h[1-6]>)", heading, html, flags=re.DOTALL)

    def link(match):
        """Resolve notebook-relative document links through the site's file map."""
        target = urlsplit(unescape(match[2]))
        if target.scheme or target.netloc or not target.path.endswith((".md", ".ipynb")):
            return match[0]
        source = posixpath.normpath(
            str(PurePosixPath(page.file.src_uri).parent / unquote(target.path))
        )
        document = files.get_file_from_path(source)
        if document is None or source == page.file.src_uri:
            return match[0]
        url = get_relative_url(document.url, page.file.url)
        url = urlunsplit(("", "", url, target.query, target.fragment))
        return match[1] + escape(url, quote=True) + match[3]

    return re.sub(r'(<a\b[^>]*\bhref=["\'])([^"\']*)(["\'])', link, html)
