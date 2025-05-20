# _plugins/last_commit_info.rb
module Jekyll
  module Git
    # Returns [short_sha, message] for the last commit touching `input_path`
    def last_commit_info(input_path)
      out = `git log -n 1 --pretty=format:%h::%s -- #{input_path}`.strip
      out.empty? ? nil : out.split("::", 2)
    end
  end
end

Liquid::Template.register_filter(Jekyll::Git)
