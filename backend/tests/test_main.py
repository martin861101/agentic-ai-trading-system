def test_read_root(client):
    """
    Test the root endpoint of the orchestrator.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Orchestrator is running"}

def test_health_check(client):
    """
    Test the health check endpoint in the API router.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
