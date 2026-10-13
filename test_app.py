"""Run: python -m unittest -v. Tests exercise real inference and security boundaries."""
import io
import re
import unittest
from pathlib import Path
import app as module


class ECGTests(unittest.TestCase):
    def setUp(self):
        module.app.config['TESTING'] = True
        module.RATE_BUCKETS.clear()
        self.original_hash = module.PASSWORD_HASH
        module.PASSWORD_HASH = None
        self.client = module.app.test_client()
        self.client.get('/')
        with self.client.session_transaction() as state:
            self.token = state['csrf']

    def tearDown(self):
        module.PASSWORD_HASH = self.original_hash

    def upload(self, content, filename='signal.csv', fs=360):
        return self.client.post('/api/analyze', data={'file': (io.BytesIO(content), filename), 'sampling_rate': str(fs)}, headers={'X-CSRF-Token': self.token})

    def test_real_model_predictions(self):
        response = self.upload((Path(module.BASE) / 'samples/mitbih_100_30s.csv').read_bytes())
        self.assertEqual(response.status_code, 200, response.json)
        result = response.json
        self.assertGreater(result['classified_beats'], 20)
        self.assertEqual(sum(result['distribution'].values()), result['classified_beats'])
        self.assertTrue(all(beat['type'] in module.CLASS_LABELS for beat in result['beats']))
        self.assertTrue(all(0 <= beat['confidence'] <= 1 for beat in result['beats']))
        self.assertGreater(result['heart_rate_bpm'], 50)
        self.assertLess(result['heart_rate_bpm'], 120)

    def test_missing_csrf(self):
        response = self.client.post('/api/analyze', data={})
        self.assertEqual(response.status_code, 403)

    def test_invalid_signals(self):
        for content, name, fs in [(b'hello world', 'bad.csv', 360), (b'1\n'*2000, 'flat.txt', 360), (b'NaN\n'*2000, 'nan.txt', 360), (b'1,2\n', 'short.csv', 360), (b'abc', 'model.joblib', 360), (b'1\n'*2000, 'bad.txt', 50)]:
            with self.subTest(name=name):
                response = self.upload(content, name, fs)
                self.assertEqual(response.status_code, 422, response.json)
                self.assertFalse(response.json['success'])
                self.assertNotIn('beats', response.json)

    def test_size_limit(self):
        response = self.upload(b'0'* (2*1024*1024+1))
        self.assertEqual(response.status_code, 413)

    def test_rate_limit(self):
        for _ in range(15):
            self.client.post('/api/analyze', headers={'X-CSRF-Token': self.token})
        response = self.client.post('/api/analyze', headers={'X-CSRF-Token': self.token})
        self.assertEqual(response.status_code, 429)

    def test_security_headers(self):
        response = self.client.get('/')
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn("script-src 'self'", response.headers['Content-Security-Policy'])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertTrue(module.app.config['SESSION_COOKIE_HTTPONLY'])

    def test_login_and_logout(self):
        module.PASSWORD_HASH = module.generate_password_hash('test-access-password')
        self.assertEqual(self.client.get('/').status_code, 302)
        self.assertEqual(self.client.get('/api/model').status_code, 401)
        self.assertEqual(self.client.post('/login', data={'csrf_token': self.token, 'password': 'wrong'}).status_code, 401)
        response = self.client.post('/login', data={'csrf_token': self.token, 'password': 'test-access-password'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get('/').status_code, 200)
        # A successful login rotates the CSRF token.
        self.assertEqual(self.client.post('/logout', data={'csrf_token': self.token}).status_code, 403)
        with self.client.session_transaction() as state:
            token = state['csrf']
        self.assertEqual(self.client.post('/logout', data={'csrf_token': token}).status_code, 302)
        self.assertEqual(self.client.get('/api/model').status_code, 401)


if __name__ == '__main__':
    unittest.main()
