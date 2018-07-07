# ISLR

Notes and exercises for James, Witten, Hastie, and Tibshirani's ["Introduction
to Statistical Learning"](http://www-bcf.usc.edu/~gareth/ISL/getbook.html)
(seventh edition).

## Development

This repo includes some convenience functions for running a containerized R
development environment.

To **launch an interactive R shell in a container**:

```
./scripts/r
```

To **compile and livereload RMarkdown files**, run the following command and
visit the development server at `http://localhost:8000`:

```
./scripts/serve
```

To **clean up unused Docker resources** once you're done developing:

```
./scripts/clean
```
