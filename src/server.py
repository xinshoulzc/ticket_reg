from digit_reg import train_main, infer_main
from http.server import HTTPServer, BaseHTTPRequestHandler
from logger import logger
import sys
import traceback
from urllib import parse

class DigitRegHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = parse.urlparse(self.path)
        param_dict = parse.parse_qs(url.query)
        inputdir = param_dict.get('inputdir',[''])[0]
        outputdir = param_dict.get('outputdir',[''])[0]
        mode = param_dict.get('mode',[''])[0]
        if len(inputdir) <= 0 or len(outputdir) <= 0 or len(mode) <= 0:
            self.response("valid params")
            return
        try:
            if mode == "price":
                infer_main(inputdir, "model", outputdir, mode)
                self.response("")
            elif mode == "barcode":
                infer_main(inputdir, "model", outputdir, mode)
                self.response("")
            else:
                self.response("UNKNOWN MODE: {}".format(mode))
        except:
            error = traceback.format_exc()
            self.response(error)

    def response(self, content):
        if len(content) <= 0:
            self.send_response(200)
            content = "success"
        else:
            self.send_response(500)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

if __name__ == '__main__':
    from http.server import HTTPServer
    server = HTTPServer(('localhost', 8081), DigitRegHandler)
    print('Starting server, use <Ctrl-C> to stop')
    try:
        server.serve_forever()
    except:
        sys.exit(-1)
