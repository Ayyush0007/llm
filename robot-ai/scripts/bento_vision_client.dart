import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class BentoVisionClient {
  final String apiUrl = 'https://sarasproject-bento.hf.space/predict';

  Future<List<Map<String, dynamic>>> detectObjects(File imageFile) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var jsonResponse = json.decode(responseData);
        return List<Map<String, dynamic>>.from(jsonResponse['detections']);
      } else {
        throw Exception('Failed to detect objects: ${response.statusCode}');
      }
    } catch (e) {
      print('Vision API Error: $e');
      return [];
    }
  }
}
