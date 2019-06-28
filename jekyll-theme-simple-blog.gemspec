# encoding: utf-8

Gem::Specification.new do |s|
  s.name          = "jekyll-theme-simple-blog"
  s.version       = "0.0.3"
  s.license       = "CC0-1.0"
  s.authors       = ["Lorenzo Pirritano"]
  s.email         = ["lorepirri@gmail.com"]
  s.homepage      = "https://github.com/lorepirri/jekyll-theme-simple-blog"
  s.summary       = "Simple Blog Theme is a clean, responsive blogging theme for Jekyll and GitHub Pages, with social/SEO, multilanguage features."

  s.files         = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^((_includes|_layouts|_sass|assets)/|(LICENSE|README|sitemaps|projects-en|projects-it|now-en|now-it|index-en|index-it|feed-en|feed-it|contact-en|contact-it|about-en|about-it|404-en|404-it)((\.(txt|md|markdown|xml)|$)))}i)
  end

  s.platform      = Gem::Platform::RUBY
  s.add_runtime_dependency "jekyll", "~> 3.6.3"
  s.add_runtime_dependency "jekyll-target-blank"
end
