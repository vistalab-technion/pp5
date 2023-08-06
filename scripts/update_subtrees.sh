#!/usr/bin/env bash

# Update the 'torustest' package using the latest version from its repo.
git subtree pull \
    --prefix src/pp5/stats/torustest \
    https://github.com/gonzalez-delgado/torustest.git \
    master \
    --squash
