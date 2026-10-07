/* MathJax 3 configuration, loaded before the tex-mml-chtml bundle.
 *
 * Arithmatex rewrites $...$ on regular Markdown pages to \(...\), but notebook
 * pages rendered by mkdocs-jupyter keep their literal $...$ delimiters, so the
 * dollar delimiters must be enabled here for both cases to typeset. */
window.MathJax = {
  tex: {
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_ignore",
  },
};
