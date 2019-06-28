![Jekyll Version](https://img.shields.io/badge/Jekyll-3.1.2-red.svg)
![Build Status](https://gitlab.com/jekyll-themes/default-bundler/badges/master/build.svg)

----

View Demo: https://lorepirri.gitlab.io/jekyll-theme-simple-blog/

-----
# Simple Blog Theme

*Simple Blog is a Jekyll theme for Gitlab or GitHub Pages. It is based on [Cayman Blog Theme](https://github.com/lorepirri/cayman-blog). You can [preview the theme to see what it looks like](https://lorepirri.gitlab.io/jekyll-theme-simple-blog/), or even [use it today](#install).*

<img src="https://gitlab.com/lorepirri/jekyll-theme-simple-blog/raw/master/simple-blog-theme.png" alt="Thumbnail of jekyll-theme-simple-blog" style="max-width:30%; border: 1px solid grey;"/>

## Features

- Blog
- Responsive
- Minimal
- Multi-language
- SEO optimized
- Social buttons (instagram, linkedin, twitter, github, gitlab)
- RSS feed multi-language

## Install

Simple Blog Theme is 100% compatible with GitLab and GitHub Pages.

### Install as a Fork

1. [Fork the repo](https://gitlab.com/lorepirri/jekyll-theme-simple-blog)
2. Clone down the repo with one of the two:
    * ssh `$ git clone git@gitlab.com:your-username/jekyll-theme-simple-blog.git`
    * https: `$ git clone https://gitlab.com/lorepirri/jekyll-theme-simple-blog.git`
3. Empty the `_posts/` folder
4. Install bundler and gems with `$ script/bootstrap`
5. Run Jekyll with `$ script/server`
6. Modify `_config.yml`, `about-en.md`, `contact-en.md`, and the other pages for your project
6. Write your posts in `_posts/en` and `_posts/<other-language>`
7. [Customize the theme](customizing)

### SEO tags

Simple Blog includes simple SEO tags from [jekyll-social-metatags](https://github.com/lorepirri/jekyll-social-metatags). Have a look at the page for its usage.

The usage is compatible with the plugin [Jekyll SEO Tag](https://github.com/jekyll/jekyll-seo-tag), which provides a battle-tested template of crowdsourced best-practices.

To switch to a better SEO tags however, one should install [Jekyll Feed plugin](https://github.com/jekyll/jekyll-feed):

1. Add this line to your site's Gemfile:

    ```ruby
    gem 'jekyll-seo-tag'
    ```

2. And then add this line to your site's `_config.yml`:

    ```yml
    gems:
      - jekyll-seo-tag
    ```

3. Replace with the following, the `<!-- jekyll-seo-tag -->` comment in your site's `default.html`:

      ```liquid
      {% seo %}
      ```

For more information about configuring this plugin, see the official [Jekyll SEO Tag](https://github.com/jekyll/jekyll-seo-tag) page.


### Stylesheet

If you'd like to add your own custom styles:

1. Create a file called `/assets/css/style.scss` in your site
2. Add the following content to the top of the file, exactly as shown:
    ```scss
    ---
    ---

    @import "{{ site.theme }}";
    ```
3. Add any custom CSS (or Sass, including imports) you'd like immediately after the `@import` line

### Layouts

If you'd like to change the theme's HTML layout:

1. [Copy the original template](https://gitlab.com/lorepirri/jekyll-theme-simple-blog/blob/master/_layouts/default.html) from the theme's repository<br />(*Pro-tip: click "raw" to make copying easier*)
2. Create a file called `/_layouts/default.html` in your site
3. Paste the default layout content copied in the first step
4. Customize the layout as you'd like

### Sass variables

If you'd like to change the theme's [Sass variables](https://gitlab.com/lorepirri/jekyll-theme-simple-blog/blob/master/_sass/variables.scss), set new values before the `@import` line in your stylesheet:

```scss
$section-headings-color: #0086b3;

@import "{{ site.theme }}";
```

## Roadmap

See the [open issues](https://gitlab.com/lorepirri/jekyll-theme-simple-blog/issues) for a list of proposed features (and known issues).

## Project philosophy

The Simple Blog Theme is intended to make it quick and easy for Gitlab or GitHub Pages users to create their first (or 100th) website. The theme should meet the vast majority of users' needs out of the box, erring on the side of simplicity rather than flexibility, and provide users the opportunity to opt-in to additional complexity if they have specific needs or wish to further customize their experience (such as adding custom CSS or modifying the default layout). It should also look great, but that goes without saying.

## Contributing

Interested in contributing to Simple Blog? We'd love your help. Simple Blog is an open source project, built one contribution at a time by users like you. See [the CONTRIBUTING file](CONTRIBUTING.md) for instructions on how to contribute.

### Previewing the theme locally

If you'd like to preview the theme locally (for example, in the process of proposing a change):

1. Clone down the theme's repository (`git clone https://gitlab.com/lorepirri/jekyll-theme-simple-blog`)
2. `cd` into the theme's directory
3. Run `script/bootstrap` to install the necessary dependencies
4. Run `script/server` to start the preview server
5. Visit [`localhost:4000`](http://localhost:4000) in your browser to preview the theme


[`.gitlab-ci.yml`]: https://gitlab.com/jekyll-themes/default-bundler/blob/master/.gitlab-ci.yml
[`Gemfile`]: https://gitlab.com/jekyll-themes/default-bundler/blob/master/Gemfile
[`.gitignore`]: https://gitlab.com/jekyll-themes/default-bundler/blob/master/.gitignore
[`_config.yml`]: https://gitlab.com/jekyll-themes/default-bundler/blob/master/_config.yml

[Bundler]: http://bundler.io/
[Jekyll]: http://jekyllrb.com/
[jek-312]: https://rubygems.org/gems/jekyll/versions/3.1.2