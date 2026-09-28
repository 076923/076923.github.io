source "https://rubygems.org"

gemspec

# Jekyll 본체 (Ruby 3.4+ / 4.x 호환 버전)
gem "jekyll", "~> 4.4"

# _config.yml 의 plugins 목록
group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-gist"
  gem "jekyll-feed"
  gem "jekyll-include-cache"
end

# Ruby 3.0 부터 표준 라이브러리에서 분리됨 (jekyll serve 에 필요)
gem "webrick", "~> 1.9"

# octokit(jekyll-gist) 의 Faraday v2 retry 미들웨어 경고 제거
gem "faraday-retry", "~> 2.3"

# Windows 환경 의존성
platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
  gem "wdm", "~> 0.2"
end
