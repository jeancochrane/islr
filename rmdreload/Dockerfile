FROM r-base:latest

RUN apt-get update && \
	apt-get install -y xdotool \
					   inotify-tools \
					   pandoc

RUN R -e "install.packages('rmarkdown')" && \
	R -e "install.packages('servr')"

COPY . /opt/rmdreload
WORKDIR /opt/rmdreload
EXPOSE 8000

ENTRYPOINT R
