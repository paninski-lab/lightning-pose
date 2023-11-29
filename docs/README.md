# ReadTheDocs set up

Welcome! This guide walks you through how to automatically generate documentation for this python project using the [sphinx](http://www.sphinx-doc.org/en/stable/index.html) package in python, and how to publish it on [Read the Docs](https://readthedocs.org/) so that users can easily access and search your documentation.

1. [Install sphinx](#1-install-sphinx)
2. [Set up conf.py file](#2-set-up-confpy-file)
3. [Build the documentation](#3-build-the-documentation)
4. [Add a new page](#4-add-a-new-page)
5. [Publish the documentation](#5-publish-the-documentation)

## 1. Install sphinx

General instructions [here](http://www.sphinx-doc.org/en/stable/tutorial.html).

Install Sphinx from the command line:

```
pip install Sphinx
```

Next we want to set up the source directory for the documentation. 
In the command line, `cd` to the root of the project documentation directory 
(something like `docs`) and enter:

```
sphinx-quickstart
```

You'll be prompted to enter a number of user options. For most you can just accept the defaults.

## 2. Set up conf.py file
Now that sphinx is installed, we want to configure it to automatically parse our meticulously maintained docstrings and generate html pages that display said information in a readable and searchable way. More details on the Google-style python docstrings used in this project can be found [here](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

### Set project path
In the `conf.py` file, uncomment the following lines at the top so that the conf file (located in `lightning_pose/docs`) can find the project (located above in `lightning_pose/`):

```python
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
```

### Theme
Change the theme to something nicer than the default. 
We chose `sphinx_rtd_theme`, which requires a pip install:

```
pip install sphinx_rtd_theme
```

Remember to add `sphix_rtd_theme` to `setup.py`.  

Then in the `conf.py` file add:

```python
html_theme = 'sphinx_rtd_theme'
import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
```

Find the themes available through sphinx [here](http://www.sphinx-doc.org/en/stable/theming.html).

### Add extensions
Expand the `extensions` variable in `conf.py`, so that it looks something like this:

```python
extensions = [
    'sphinx.ext.autodoc', # allows automatic parsing of docstrings
    'sphinx.ext.mathjax', # allows mathjax in documentation
    'sphinx.ext.viewcode', # links documentation to source code
    'sphinx.ext.githubpages', # allows integration with github
    'sphinx.ext.napoleon' # parsing of different docstring styles
    'sphinx_automodapi.automodapi',
    'sphinx_copybutton',  # add copy button to code blocks
] 
```

You'll need to install `automodapi` and `copybutton` and add them to your requirements in 
`setup.py`:

```
pip install sphinx-automodapi sphinx-copybutton
```

### Include documentation for class constructors
If you want to document `__init__()` functions for python classes, add the following functions to the end of the `conf.py` file (thanks to [https://stackoverflow.com/a/5599712]()):

```python
def skip(app, what, name, obj, skip, options):
    if name == '__init__':
        return False
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip)
```

### Automatic API documentation

In order to make the `sphinx-automodapi` work, you will need to create a `modules` directory; 
see [here](https://sphinx-automodapi.readthedocs.io/en/latest/)
and `lightning_pose/docs/modules` for examples.

In the codebase, you will also need to add a list at the top of each module that lists the 
functions and classes to be included in the docs, 
otherwise imported functions will be documented as well:

```python
__all__ = [
    "function_0",
    "function_1",
    "Class0",
]
```

## 3. Build the documentation
In the `docs` directory, run:
```
make html
```

You'll then be able to find the documentation landing page at `docs/_build/html/index.html`

## 4. Add a new page
Docstrings are useful for understanding how individual functions work, but do not help too much 
for a new user of the code. 
To facilitate learning how the code works we will want to create tutorial pages that demonstrate 
how to use certain features of the code.

In the directory containing the `index.rst` file, add a new file called `tutorial-example.rst`. 
This will look something like:

```
################
Tutorial Example
################

Here's some content written in reStructured Text (.rst), a markup language 
commonly used for technical documentation
```

Tell sphinx where this file is by adding `tutorial-example` to the `.. toctree::` section in the 
`index.rst` file, so that it looks something like this:

```
.. toctree::
   :maxdepth: 2

   tutorial-example
   another-tutorial-example
```

To see the updated html pages you need to rebuild the documentation:
```
make html
```

## 5. Publish the documentation 

General instructions 
[here](http://dont-be-afraid-to-commit.readthedocs.io/en/latest/documentation.html).

Now that we've built our documentation, we want to publish it on the web. 
Fortunately, ReadTheDocs (RTD) and GitHub make this super simple. 
The following steps are mostly copy-and-pasted from the general instructions above.

### Exclude unwanted directories
We do not want to commit the rendered files to github, just the source. To exclude these, add them to `.gitignore`:

```
_build
_static
_templates
```

Then push the updated files to GitHub.

### Create an account with readthedocs.org
Follow the instructions there, they should be self-explanatory.

And now, just like magic, RTD will watch your GitHub project and update the documentation every night.

But wait! You can do better, if you really think it is necessary. On GitHub:
1. select **settings** for your project (not for your account!) in the navigation panel on the right-hand side
2. choose **Webhooks & Services**
3. enable `ReadTheDocs` under **Add Service** dropdown

...and now, every time you push documents to GitHub, RTD will be informed that you have new documents to be published. It's not magic, they say, but it's pretty close. Close enough for me.
